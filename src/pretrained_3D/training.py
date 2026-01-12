import argparse
import os
import random
from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from egnn import EGNN
import math

from dataloader import PT3DDataset
from utils import collate_views_rdkit_bonds, atoms_to_Z, collate_views_rdkit_bonds_val

RDLogger.DisableLog("rdApp.*")


def split_indices(n, val_frac=0.1, seed=33):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = int(round(n * val_frac))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx

class EGNNContrastiveEncoder(nn.Module):
    """
    egnn.forward(h, x, edges, edge_attr) -> (node_emb, coords)
    - h: (total_nodes, atom_emb_dim)
    - x: (total_nodes, 3)
    - edges: (2, total_edges)
    - edge_attr: (total_edges, in_edge_nf=13)
    """
    def __init__(self, egnn, atom_emb_dim=64, proj_dim=128):
        super().__init__()
        self.egnn = egnn
        self.atom_emb = nn.Embedding(119, atom_emb_dim)  # Z in [0..118]
        self.proj = nn.Linear(egnn.embedding_out.out_features, proj_dim)

    def forward(self, Z, x, edges, edge_attr, batch_idx):
        h0 = self.atom_emb(Z)                # (T, atom_emb_dim)
        h, _ = self.egnn(h0, x, edges, edge_attr)  # h: (T, out_node_nf)

        # mean pooling by molecule (batch_idx)
        B = int(batch_idx.max().item()) + 1
        hdim = h.size(1)

        sum_h = h.new_zeros((B, hdim))
        cnt = h.new_zeros((B, 1))

        sum_h.scatter_add_(0, batch_idx.unsqueeze(-1).expand(-1, hdim), h)
        ones = torch.ones((h.size(0), 1), device=h.device, dtype=h.dtype)
        cnt.scatter_add_(0, batch_idx.unsqueeze(-1), ones)

        z = sum_h / cnt.clamp(min=1.0)

        z = self.proj(z)
        z = F.normalize(z, dim=-1)
        return z


def info_nce(z1, z2, tau=0.1, symmetric=True):
    """
    z1,z2: (B,d), already normalized
    """
    logits = (z1 @ z2.t()) / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    loss12 = F.cross_entropy(logits, labels)
    if not symmetric:
        return loss12
    loss21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss12 + loss21)


def build_collate_train(args):
    def _fn(batch, atoms_to_Z, device):
        # uses stochastic augmentations
        return collate_views_rdkit_bonds(
            batch=batch,
            atoms_to_Z=atoms_to_Z,
            device=device,
            rotate=bool(args.rotate),
            noise_std=float(args.noise_std),
            edge_drop=float(args.edge_drop),
            drop_if_no_bonds=bool(args.drop_if_no_bonds),
        )
    return _fn


def build_collate_val(args):
    def _fn(batch, atoms_to_Z, device):
        # deterministic: NO noise, NO edge_drop, fixed rotations inside collate_views_rdkit_bonds_val
        return collate_views_rdkit_bonds_val(
            batch=batch,
            atoms_to_Z=atoms_to_Z,
            device=device,
            drop_if_no_bonds=bool(args.drop_if_no_bonds),
        )
    return _fn


@torch.no_grad()
def eval_contrastive_loss(model, loader, collate_val, atoms_to_Z, device, tau: float):
    model.eval()
    losses = []
    for batch in loader:
        v1, v2, _ = collate_val(batch, atoms_to_Z=atoms_to_Z, device=device)
        Z1, x1, e1, ea1, b1 = v1
        Z2, x2, e2, ea2, b2 = v2

        B = int(b1.max().item()) + 1
        if B < 2:
            continue

        z1 = model(Z1, x1, e1, ea1, b1)
        z2 = model(Z2, x2, e2, ea2, b2)

        loss = info_nce(z1, z2, tau=tau, symmetric=True)
        losses.append(float(loss.item()))

    return sum(losses) / max(1, len(losses))

def train_egnn_contrastive_pt(
    pt_path: str,
    egnn,
    atoms_to_Z,
    out_ckpt: str,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    tau: float = 0.1,
    proj_dim: int = 128,
    atom_emb_dim: int = 64,
    grad_clip: float = 1.0,
    num_workers: int = 0,
    device: str = "cuda",
    amp: bool = True,
    seed: int = 33,

    # pretrain-val
    val_frac: float = 0.1,
    patience: int = 15,
    min_delta: float = 1e-4,

    # collates
    collate_train=None,
    collate_val=None,
    shuffle_train: bool = True,
    save_last: bool = True,
):
    ds = PT3DDataset(
        pt_path,
        remove_multifragment=False,
        min_atoms=3,
        max_atoms=256,
        drop_all_zeros=True,
        drop_z_all_zeros=False,
        seed=seed,
        save_valid_ids=None
    )

    train_idx, val_idx = split_indices(len(ds), val_frac=val_frac, seed=seed)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=lambda b: b,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,                 # IMPORTANT: deterministic validation
        num_workers=num_workers,
        collate_fn=lambda b: b,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )

    model = EGNNContrastiveEncoder(egnn, atom_emb_dim=atom_emb_dim, proj_dim=proj_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.startswith("cuda")))

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0

    out_dir = os.path.dirname(out_ckpt) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_last = os.path.join(out_dir, "pretrained_egnn_contrastive_last.pt")

    print(f"[Data] total={len(ds)} train={len(train_ds)} val={len(val_ds)} val_frac={val_frac}")
    print(f"[Device] {device} | AMP={scaler.is_enabled()} | shuffle_train={shuffle_train}")

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        used = 0

        for bi, batch in enumerate(train_loader, start=1):
            try:
                v1, v2, _ = collate_train(batch, atoms_to_Z=atoms_to_Z, device=device)
            except Exception:
                continue

            Z1, x1, e1, ea1, b1 = v1
            Z2, x2, e2, ea2, b2 = v2

            B = int(b1.max().item()) + 1
            if B < 2:
                continue

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                z1 = model(Z1, x1, e1, ea1, b1)
                z2 = model(Z2, x2, e2, ea2, b2)
                loss = info_nce(z1, z2, tau=tau, symmetric=True)

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.item()))
            used += 1

            # if log_every and (used % log_every == 0):
            #     avg = sum(losses[-log_every:]) / max(1, len(losses[-log_every:]))
            #     print(f"[ep {ep:03d}/{epochs}] step {used:04d} train_loss {avg:.4f} B={B}")

        train_loss = sum(losses) / max(1, len(losses))

        # deterministic validation
        val_loss = eval_contrastive_loss(
            model=model,
            loader=val_loader,
            collate_val=collate_val,
            atoms_to_Z=atoms_to_Z,
            device=device,
            tau=tau,
        )

        print(f"[ep {ep:03d}/{epochs}] train={train_loss:.4f} | val(det)={val_loss:.4f} | used_batches={used}")

        if save_last:
            torch.save({
                "epoch": ep,
                "best_val": best_val,
                "encoder": model.state_dict(),
                "egnn": egnn.state_dict(),
                "optimizer": opt.state_dict(),
                "config": {
                    "pt_path": pt_path,
                    "val_frac": val_frac,
                    "seed": seed,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "tau": tau,
                    "proj_dim": proj_dim,
                    "atom_emb_dim": atom_emb_dim,
                    "in_edge_nf": getattr(egnn, "in_edge_nf", None),
                }
            }, out_last)

        # best ckpt + early stop
        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_epoch = ep
            bad_epochs = 0

            torch.save({
                "epoch": ep,
                "best_val": best_val,
                "encoder": model.state_dict(),
                "egnn": egnn.state_dict(),
                "optimizer": opt.state_dict(),
                "config": {
                    "pt_path": pt_path,
                    "val_frac": val_frac,
                    "seed": seed,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "tau": tau,
                    "proj_dim": proj_dim,
                    "atom_emb_dim": atom_emb_dim,
                    "in_edge_nf": getattr(egnn, "in_edge_nf", None),
                }
            }, out_ckpt)

            print(f"  -> NEW BEST saved: {out_ckpt} (val={best_val:.4f} @ ep={ep})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stop: no val improvement for {patience} epochs. "
                      f"Best ep={best_epoch}, best val={best_val:.4f}")
                break

    return model, ds


def parse_args():
    p = argparse.ArgumentParser("EGNN contrastive pretrain from .pt (raw confs)")

    # I/O
    p.add_argument("--pt_path", type=str, required=True, default="/content/raw_confs.pt")
    p.add_argument("--out_ckpt", type=str, default="/content/pretrained_egnn_contrastive_best.pt")
    p.add_argument("--save_last", action="store_true", help="Also save last checkpoint each epoch")

    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--no_shuffle", action="store_true")

    # Pretrain-val (deterministic)
    p.add_argument("--val_frac", type=float, default=0.1, help="Holdout fraction for pretrain validation (0.05-0.10 recommended)")
    p.add_argument("--patience", type=int, default=15, help="Early stop patience on val loss")
    p.add_argument("--min_delta", type=float, default=1e-4, help="Min val improvement to reset patience")

    # EGNN config
    p.add_argument("--atom_emb_dim", type=int, default=64)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--hidden_nf", type=int, default=128)
    p.add_argument("--out_node_nf", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--attention", type=int, default=0)
    p.add_argument("--normalize", type=int, default=0)
    p.add_argument("--tanh", type=int, default=0)

    # Edge attr
    p.add_argument("--in_edge_nf", type=int, default=13)

    # Augmentations for TRAIN views only
    p.add_argument("--rotate", type=int, default=1)
    p.add_argument("--noise_std", type=float, default=0.02)
    p.add_argument("--edge_drop", type=float, default=0.1)
    p.add_argument("--drop_if_no_bonds", type=int, default=1)

    # Misc
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--device", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    # device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # set_seed(args.seed, deterministic=True)

    # Init EGNN
    egnn = EGNN(
        in_node_nf=args.atom_emb_dim,
        hidden_nf=args.hidden_nf,
        out_node_nf=args.out_node_nf,
        in_edge_nf=args.in_edge_nf,
        device=device,
        n_layers=args.n_layers,
        attention=bool(args.attention),
        normalize=bool(args.normalize),
        tanh=bool(args.tanh),
    ).to(device)

    # collates
    collate_train = build_collate_train(args)
    collate_val = build_collate_val(args)

    # Train
    model, ds = train_egnn_contrastive_pt(
        pt_path=args.pt_path,
        egnn=egnn,
        atoms_to_Z=atoms_to_Z,
        out_ckpt=args.out_ckpt,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        tau=args.tau,
        proj_dim=args.proj_dim,
        atom_emb_dim=args.atom_emb_dim,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        device=device,
        amp=(not args.no_amp),
        seed=args.seed,
        val_frac=args.val_frac,
        patience=args.patience,
        min_delta=args.min_delta,
        collate_train=collate_train,
        collate_val=collate_val,
        shuffle_train=(not args.no_shuffle),
        save_last=args.save_last,
    )

    print(f"Done. Best ckpt saved to: {args.out_ckpt}")
    print(f"Dataset total used     : {len(ds)}")
    print(f"Device                : {device}")


if __name__ == "__main__":
    main()