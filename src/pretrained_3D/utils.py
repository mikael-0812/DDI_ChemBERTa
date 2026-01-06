import math

import numpy as np
import torch
from typing import Dict, Any, Tuple

from rdkit import Chem

pt = Chem.GetPeriodicTable()
def entry_to_coords_pt(entry: Dict[str, Any]) -> np.ndarray:
    """
    entry['confs'] is list length N_atoms, each element np.ndarray shape (3,)
    Return coords np.float32 [N,3]
    """
    coords = np.asarray(entry["confs"], dtype=np.float32)
    if coords.ndim == 2 and coords.shape[1] == 3:
        return coords
    raise ValueError(f"Unexpected entry['confs'] -> coords shape: {coords.shape}")

def atoms_to_Z_symbols(atoms) -> np.ndarray:
    return np.array([pt.GetAtomicNumber(a) for a in atoms], dtype=np.int64)

def check_coords_fail(coords: np.ndarray, n_expected: int, allow_2d: bool = False) -> str:
    if coords.ndim != 2 or coords.shape[1] != 3:
        return "bad_shape"
    if coords.shape[0] != n_expected:
        return "atom_mismatch"
    if not np.isfinite(coords).all():
        return "nan_inf"
    if np.all(coords == 0.0):
        return "all_zero"
    if (not allow_2d) and np.all(coords[:, 2] == 0.0):
        return "z_all_zero"
    return "ok"

def build_radius_graph_single(
    x: torch.Tensor, r_cut: float = 4.5, return_edge_attr: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: [N,3]
    edges: [2,E], edge_attr: [E,1] (r^2) if return_edge_attr else None
    """
    N = x.size(0)
    diff = x[:, None, :] - x[None, :, :]
    dist2 = (diff * diff).sum(dim=-1)

    mask = (dist2 <= (r_cut ** 2)) & (~torch.eye(N, dtype=torch.bool, device=x.device))
    row, col = mask.nonzero(as_tuple=True)

    edges = torch.stack([row, col], dim=0)  # [2,E]
    if return_edge_attr:
        edge_attr = dist2[row, col].unsqueeze(-1)  # [E,1]
    else:
        edge_attr = None
    return edges, edge_attr

def confs_to_coords(confs):
    coords = np.asarray(confs, dtype=np.float32)
    if coords.ndim == 2 and coords.shape[1] == 3:
        return coords
    raise ValueError(f"bad coords shape: {coords.shape}")


def random_rotation_matrix_torch(device):
    A = torch.randn(3, 3, device=device)
    Q, _ = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def build_radius_edges_only(x, r_cut=4.5):
    """
    x: [N,3] -> edges [2,E] (NO edge_attr)
    """
    N = x.size(0)
    diff = x[:, None, :] - x[None, :, :]
    dist2 = (diff * diff).sum(dim=-1)
    mask = (dist2 <= (r_cut**2)) & (~torch.eye(N, dtype=torch.bool, device=x.device))
    row, col = mask.nonzero(as_tuple=True)
    return torch.stack([row, col], dim=0)

def edge_dropout_edges(edges, drop_p=0.1):
    if drop_p <= 0 or edges.numel() == 0:
        return edges
    E = edges.size(1)
    keep = torch.rand(E, device=edges.device) > drop_p
    return edges[:, keep]

def collate_two_views_edges_only(
    batch, device="cuda", r_cut=4.5, noise_std=0.02, edge_drop=0.1, rotate=True
):
    """
    batch items: (key_int, smiles, atoms, confs)
    return:
      view1: (Z, x, edges, batch_idx)
      view2: ...
      keys: list[int]
    """
    keys = []

    Z1_all, x1_all, b1_all, e1_all = [], [], [], []
    Z2_all, x2_all, b2_all, e2_all = [], [], [], []

    offset1 = 0
    offset2 = 0

    for bi, (k, smiles, atoms, confs) in enumerate(batch):
        keys.append(int(k))

        coords = confs_to_coords(confs)
        Z = atoms_to_Z_symbols(atoms)

        x0 = torch.tensor(coords, dtype=torch.float32, device=device)
        x0 = x0 - x0.mean(dim=0, keepdim=True)
        Zt = torch.tensor(Z, dtype=torch.long, device=device)

        # ---- view1 ----
        x1 = x0.clone()
        if rotate:
            R = random_rotation_matrix_torch(device)
            x1 = x1 @ R.T
        if noise_std > 0:
            x1 = x1 + noise_std * torch.randn_like(x1)

        e1 = build_radius_edges_only(x1, r_cut=r_cut)
        e1 = edge_dropout_edges(e1, drop_p=edge_drop)
        if e1.numel() == 0:
            continue
        e1 = e1 + offset1

        Z1_all.append(Zt)
        x1_all.append(x1)
        b1_all.append(torch.full((x1.size(0),), bi, device=device, dtype=torch.long))
        e1_all.append(e1)
        offset1 += x1.size(0)

        # ---- view2 ----
        x2 = x0.clone()
        if rotate:
            R = random_rotation_matrix_torch(device)
            x2 = x2 @ R.T
        if noise_std > 0:
            x2 = x2 + noise_std * torch.randn_like(x2)

        e2 = build_radius_edges_only(x2, r_cut=r_cut)
        e2 = edge_dropout_edges(e2, drop_p=edge_drop)
        if e2.numel() == 0:
            continue
        e2 = e2 + offset2

        Z2_all.append(Zt)
        x2_all.append(x2)
        b2_all.append(torch.full((x2.size(0),), bi, device=device, dtype=torch.long))
        e2_all.append(e2)
        offset2 += x2.size(0)

    def pack(Z_all, x_all, b_all, e_all):
        Zc = torch.cat(Z_all, dim=0)
        xc = torch.cat(x_all, dim=0)
        bc = torch.cat(b_all, dim=0)
        ec = torch.cat(e_all, dim=1)
        return Zc, xc, ec, bc

    v1 = pack(Z1_all, x1_all, b1_all, e1_all)
    v2 = pack(Z2_all, x2_all, b2_all, e2_all)
    return v1, v2, keys
