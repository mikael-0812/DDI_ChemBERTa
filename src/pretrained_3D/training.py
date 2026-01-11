from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from egnn import EGNN
import math

from dataloader import PT3DDataset
from utils import collate_views_rdkit_bonds, atoms_to_Z

RDLogger.DisableLog("rdApp.*")

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

def train_contrastive_from_pt(
    pt_path,
    egnn,
    atoms_to_Z,
    collate_fn,
    out_ckpt="/content/pretrained_egnn_contrastive.pt",
    batch_size=32,
    epochs=50,
    lr=1e-3,
    weight_decay=1e-6,
    device=None,
    tau=0.1,
    proj_dim=128,
    atom_emb_dim=64,
    grad_clip=1.0,
    log_every=20,
    num_workers=0,
    shuffle=True,
    drop_all_zero=True,
    drop_z_all_zero=False,
    amp=True,                   # mixed precision on Colab
):
    """
    - Loads raw dict .pt
    - Creates 2 augmented views per molecule
    - Trains EGNN + atom_emb + projection head using InfoNCE
    - Saves checkpoint with model + optimizer + config

    Requirements:
      - egnn: initialized with in_node_nf == atom_emb_dim, in_edge_nf == 13
      - collate_fn must return: v1, v2, ids
            v = (Z, x, edges, edge_attr, batch_idx)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = PT3DDataset(pt_path,
        remove_multifragment = False, # salts/ions multi-fragments
        min_atoms = 3, # removes drug molecules including only salts and metal ions
        max_atoms = 256,
        drop_all_zeros = True,
        drop_z_all_zeros = False,
        seed = 33,
        save_valid_ids = None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                        collate_fn=lambda b: b, pin_memory=(device.startswith("cuda")))

    model = EGNNContrastiveEncoder(egnn, atom_emb_dim=atom_emb_dim, proj_dim=proj_dim).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.startswith("cuda")))

    print(f"Loaded dataset: {len(ds)} molecules from {pt_path}")
    print(f"Device: {device} | AMP: {scaler.is_enabled()}")
    print(f"Saving to: {out_ckpt}")

    step = 0
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        kept_batches = 0

        for batch in loader:

            step += 1
            try:
                v1, v2, ids = collate_fn(
                    batch,
                    atoms_to_Z=atoms_to_Z,
                    device=device,
                )
            except Exception as ex:
                # batch could become empty after filtering inside collate
              print("COLLATE FAIL:", type(ex).__name__, str(ex)[:200])
              continue


            Z1, x1, e1, ea1, b1 = v1
            Z2, x2, e2, ea2, b2 = v2

            # guard: need at least 2 samples for contrastive
            B = int(b1.max().item()) + 1
            if B < 2:
                continue

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                z1 = model(Z1, x1, e1, ea1, b1)
                z2 = model(Z2, x2, e2, ea2, b2)
                loss = info_nce(z1, z2, tau=tau, symmetric=True)

            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.item()))
            kept_batches += 1

            # if log_every and (kept_batches % log_every == 0):
            #     avg = sum(losses[-log_every:]) / max(1, len(losses[-log_every:]))
            #     print(f"[ep {ep:03d}/{epochs}] batch {kept_batches:04d} | loss {avg:.4f} | B={B}")

        ep_loss = sum(losses) / max(1, len(losses))
        print(f"Epoch {ep}/{epochs} - mean loss: {ep_loss:.4f} | used_batches={kept_batches}")

        # save every epoch
        ckpt = {
            "epoch": ep,
            "encoder": model.state_dict(),
            "egnn": egnn.state_dict(),
            "optimizer": opt.state_dict(),
            "config": {
                "pt_path": pt_path,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "tau": tau,
                "proj_dim": proj_dim,
                "atom_emb_dim": atom_emb_dim,
                "in_edge_nf": getattr(egnn, "in_edge_nf", None),
            },
        }
        torch.save(ckpt, out_ckpt)

    return model, ds

def collate_fn(batch, atoms_to_Z, device):
    return collate_views_rdkit_bonds(
        batch=batch,
        atoms_to_Z=atoms_to_Z,
        device=device,
        rotate=True,
        noise_std=0.02,
        edge_drop=0.1,
        drop_if_no_bonds=True,
    )


@torch.no_grad()
def export_embeddings_from_pt_with_contrastive_encoder(
    pt_path: str,
    encoder_model,                 # EGNNContrastiveEncoder (đã train)
    out_path: str = "/content/drug_embedding.pt",
    r_cut: float = 4.5,
    batch_size: int = 64,
    allow_2d: bool = False,
    which: str = "preproj",        # "preproj" hoặc "postproj"
    normalize_postproj: bool = True
):
    """
    Input: .pt dict[int -> {'smiles','atoms','confs'}]
    Output:
      - out_path: dict[int -> embedding tensor]
      - out_path.replace(".pt","_failed.pt"): dict[int -> reason]
    """

    device = next(encoder_model.parameters()).device
    encoder_model.eval()

    ds = PT3DDataset(pt_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda b: b)

    emb = {}
    failed = {}


    for batch in loader:
        # build a batched graph (concat nodes, offset edges)
        Z_all, x_all, b_all, e_all = [], [], [], []
        keys_kept = []
        node_offset = 0
        batch_id_running = 0

        for (k, smiles, atoms, confs) in batch:
            # coords
            try:
                coords = confs_to_coords(confs)  # (N,3)
            except Exception:
                failed[int(k)] = "parse_error"
                continue

            status = check_coords_fail(coords, n_expected=len(atoms))
            if status != "ok":
                failed[int(k)] = status
                continue

            # Z
            try:
                Z = atoms_to_Z(atoms)  # (N,)
            except Exception:
                failed[int(k)] = "bad_atoms"
                continue

            x = torch.tensor(coords, dtype=torch.float32, device=device)
            x = x - x.mean(dim=0, keepdim=True)
            Zt = torch.tensor(Z, dtype=torch.long, device=device)

            edges = build_radius_edges_only(x, r_cut=r_cut)  # [2,E]
            if edges.numel() == 0:
                failed[int(k)] = "no_edges"
                continue

            edges = edges + node_offset

            Z_all.append(Zt)
            x_all.append(x)
            b_all.append(torch.full((x.size(0),), batch_id_running, device=device, dtype=torch.long))
            e_all.append(edges)

            keys_kept.append(int(k))
            node_offset += x.size(0)
            batch_id_running += 1

        if len(keys_kept) == 0:
            continue

        Z_cat = torch.cat(Z_all, dim=0)
        x_cat = torch.cat(x_all, dim=0)
        b_cat = torch.cat(b_all, dim=0)
        e_cat = torch.cat(e_all, dim=1)

        h0 = encoder_model.atom_emb(Z_cat)                         # [total_nodes, atom_emb_dim]
        h, _ = encoder_model.egnn(h0, x_cat, e_cat, edge_attr=None)  # [total_nodes, out_node_nf]

        B = int(b_cat.max().item()) + 1
        hdim = h.size(1)
        sum_h = h.new_zeros((B, hdim))
        cnt = h.new_zeros((B, 1))
        sum_h.scatter_add_(0, b_cat.unsqueeze(-1).expand(-1, hdim), h)
        ones = torch.ones((b_cat.size(0), 1), device=device, dtype=cnt.dtype)
        cnt.scatter_add_(0, b_cat.unsqueeze(-1), ones)
        z_pre = sum_h / cnt.clamp(min=1.0)                         # [B, out_node_nf]

        if which == "preproj":
            z = z_pre
        elif which == "postproj":
            z = encoder_model.proj(z_pre)
            if normalize_postproj:
                z = torch.nn.functional.normalize(z, dim=-1)
        else:
            raise ValueError("which must be 'preproj' or 'postproj'")

        z = z.detach().cpu()

        # map back
        for i, k in enumerate(keys_kept):
            emb[k] = z[i]

    torch.save(emb, out_path)
    torch.save(failed, out_path.replace(".pt", "_failed.pt"))
    print(f"Saved embeddings: {len(emb)} -> {out_path}")
    print(f"Saved failed    : {len(failed)} -> {out_path.replace('.pt','_failed.pt')}")
    return emb, failed



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    egnn = EGNN(in_node_nf=64, hidden_nf=128, out_node_nf=128, in_edge_nf=13).to(device)

    model, ds = train_contrastive_from_pt(
        pt_path="/content/raw_confs.pt",
        egnn=egnn,
        atoms_to_Z=atoms_to_Z,
        collate_fn=collate_fn,
        out_ckpt="/content/pretrained_egnn_contrastive.pt",
        batch_size=32,
        epochs=100,
        lr=1e-3,
        device=device,
        tau=0.1,
    )

    # emb, failed = export_embeddings_from_pt_with_contrastive_encoder(
    #     pt_path="/content/.pt",
    #     encoder_model=model,  # model đã train contrastive
    #     out_path="/content/drug_embedding_preproj.pt",
    #     r_cut=4.5,
    #     batch_size=64,
    #     which="preproj",
    # )
