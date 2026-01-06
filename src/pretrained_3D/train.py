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
from utils import atoms_to_Z_symbols, check_coords_fail, entry_to_coords_pt, build_radius_graph_single, \
    collate_two_views_edges_only, confs_to_coords, build_radius_edges_only

RDLogger.DisableLog("rdApp.*")

class EGNNContrastiveEncoder(nn.Module):
    def __init__(self, egnn, atom_emb_dim=64, proj_dim=128):
        super().__init__()
        self.egnn = egnn
        self.atom_emb = nn.Embedding(119, atom_emb_dim)
        self.proj = nn.Linear(egnn.embedding_out.out_features, proj_dim)

    def forward(self, Z, x, edges, batch_idx):
        h0 = self.atom_emb(Z)
        h, _ = self.egnn(h0, x, edges, edge_attr=None)  # <- edge_attr None

        B = int(batch_idx.max().item()) + 1
        hdim = h.size(1)
        sum_h = h.new_zeros((B, hdim))
        cnt = h.new_zeros((B, 1))
        sum_h.scatter_add_(0, batch_idx.unsqueeze(-1).expand(-1, hdim), h)
        ones = torch.ones((batch_idx.size(0), 1), device=batch_idx.device, dtype=cnt.dtype)
        cnt.scatter_add_(0, batch_idx.unsqueeze(-1), ones)
        z = sum_h / cnt.clamp(min=1.0)

        z = self.proj(z)
        z = F.normalize(z, dim=-1)
        return z

def info_nce(z1, z2, tau=0.1):
    logits = (z1 @ z2.t()) / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)

def train_contrastive_from_pt_edges_only(
    pt_path,
    egnn,
    out_ckpt="/content/pretrained_egnn.pt",
    batch_size=32,
    epochs=100,
    lr=1e-3,
    device=None,
    r_cut=4.5,
    noise_std=0.02,
    edge_drop=0.1,
    tau=0.1,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = PT3DDataset(pt_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=lambda b: b)

    model = EGNNContrastiveEncoder(egnn).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            v1, v2, keys = collate_two_views_edges_only(
                batch, device=device, r_cut=r_cut, noise_std=noise_std, edge_drop=edge_drop, rotate=True
            )
            Z1, x1, e1, b1 = v1
            Z2, x2, e2, b2 = v2

            z1 = model(Z1, x1, e1, b1)
            z2 = model(Z2, x2, e2, b2)

            loss = 0.5 * (info_nce(z1, z2, tau=tau) + info_nce(z2, z1, tau=tau))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.item()))

        print(f"Epoch {ep}/{epochs} - loss: {sum(losses)/max(len(losses),1):.4f}")

    torch.save({"encoder": model.state_dict()}, out_ckpt)
    return model

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
                Z = atoms_to_Z_symbols(atoms)  # (N,)
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

        # ----- forward encoder (manual để lấy preproj nếu cần) -----
        h0 = encoder_model.atom_emb(Z_cat)                         # [total_nodes, atom_emb_dim]
        h, _ = encoder_model.egnn(h0, x_cat, e_cat, edge_attr=None)  # [total_nodes, out_node_nf]

        # mean pool theo batch index
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

    egnn = EGNN(in_node_nf=64, hidden_nf=128, out_node_nf=128, in_edge_nf=0).to(device)

    model = train_contrastive_from_pt_edges_only(
        pt_path="/content/.pt",
        egnn=egnn,
        out_ckpt="/content/pretrained_egnn.pt",
        batch_size=32,
        epochs=100,
        lr=1e-3,
        device=device,
        r_cut=4.5,
        noise_std=0.02,
        edge_drop=0.1,
        tau=0.1,
    )

    emb, failed = export_embeddings_from_pt_with_contrastive_encoder(
        pt_path="/content/.pt",
        encoder_model=model,  # model đã train contrastive
        out_path="/content/drug_embedding_preproj.pt",
        r_cut=4.5,
        batch_size=64,
        which="preproj",
    )
