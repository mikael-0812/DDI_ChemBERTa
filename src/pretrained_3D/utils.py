import math

import numpy as np
import torch
from typing import Dict, Any
from collections import Counter
from numpy import bool_
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

def atoms_to_Z(atoms):
    """
    Atoms mapping to Z
    :param atoms:
    :return: (int64) array representation of atoms
    """

    Z = []
    for atom in atoms:
        z = pt.GetAtomicNumber(str(atom))
        if z <= 0:
            raise ValueError(f"Unexpected atom: {atom}")
        Z.append(z)

    return np.array(Z, dtype=np.int64)

def is_all_zeros(coords: np.ndarray) -> bool:
    return bool(np.all(coords == 0.0))

def is_z_all_zeros(coords: np.ndarray) -> bool:
    return coords.ndim == 3 and coords.shape[1] == 3 and np.all(coords[:, 2] == 0.0)


def random_rotation_matrix(device):
    """
    Return random rotation matrix for augmentation (rotation + noise)
    :param device:
    :return:
    """

    u1 = torch.rand((), device=device)
    u2 = torch.rand((), device=device)
    u3 = torch.rand((), device=device)

    q1 = torch.sqrt(1-u1) * torch.sin(2*math.pi*u2)
    q2 = torch.sqrt(1-u1) * torch.cos(2*math.pi*u2)
    q3 = torch.sqrt(u1)   * torch.sin(2*math.pi*u3)
    q4 = torch.sqrt(u1)   * torch.cos(2*math.pi*u3)

    # Rotation matrix from quaternion
    R = torch.stack([
        torch.stack([1-2*(q3*q3+q4*q4), 2*(q2*q3-q1*q4),   2*(q2*q4+q1*q3)]),
        torch.stack([2*(q2*q3+q1*q4),   1-2*(q2*q2+q4*q4), 2*(q3*q4-q1*q2)]),
        torch.stack([2*(q2*q4-q1*q3),   2*(q3*q4+q1*q2),   1-2*(q2*q2+q3*q3)]),
    ], dim=0)
    return R

def augment_coords(x, rotate=True, noise_std=0.02):
    """
    Function to augment coordinates if z_all_zeros
    :param x:
    :param rotate:
    :param noise_std:
    :return:
    """

    if rotate:
        R = random_rotation_matrix(x.device)
        x = x @ R.T

        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std

        return x

def one_hot(x, choices):
    out = [0.0] * len(choices)
    if x in choices:
        out[choices.index(x)] = 1.0
    return out

def build_bond_graph_rdkit(smiles: str, device="cuda"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    stereo_types = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]

    rows, cols, feats = [], [], []

    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()

        f = []
        f += [1.0]  # is_bond
        f += one_hot(b.GetBondType(), bond_types)         # 4
        f += [float(b.GetIsConjugated())]                 # 1
        f += [float(b.IsInRing())]                        # 1
        f += one_hot(b.GetStereo(), stereo_types)         # 6

        if len(f) != 13:
            raise RuntimeError(f"edge_attr dim != 13 for smiles={smiles}")

        rows += [i, j]
        cols += [j, i]
        feats += [f, f]

    if len(rows) == 0:
        edges = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.zeros((0, 13), dtype=torch.float32, device=device)
        return edges, edge_attr, mol.GetNumAtoms()

    edges = torch.tensor([rows, cols], dtype=torch.long, device=device)
    edge_attr = torch.tensor(feats, dtype=torch.float32, device=device)
    return edges, edge_attr, mol.GetNumAtoms()


def drop_edges(edges, edge_attr, drop_p=0.1):
    """
    edges: (2,E), edge_attr: (E,13)
    returns subsampled edges/edge_attr
    """
    if drop_p <= 0 or edges.size(1) == 0:
        return edges, edge_attr

    E = edges.size(1)
    keep = torch.rand(E, device=edges.device) > drop_p
    # bảo đảm không drop hết
    if keep.sum() == 0:
        keep[torch.randint(0, E, (1,), device=edges.device)] = True
    return edges[:, keep], edge_attr[keep]


def collate_views_rdkit_bonds(
    batch,
    atoms_to_Z,
    device='cuda',
    rotate=True,
    noise_std=0.02,
    edge_drop=0.1,
    drop_if_no_bonds=False,
    r_cut=4.5,
):
    Z1_list, x1_list, b1_list, e1_list, ea1_list = [], [], [], [], []
    Z2_list, x2_list, b2_list, e2_list, ea2_list = [], [], [], [], []

    ids = []
    node_offset = 0
    mol_idx = 0
    drop = Counter()

    for entry in batch:
        smiles = entry.get("smiles", None)
        atoms  = entry.get("atoms", None)
        coords = entry.get("confs", None)

        if smiles is None or atoms is None:
            drop["missing_smiles_or_atoms"] += 1
            continue


        x0 = torch.tensor(coords, dtype=torch.float32, device=device)
        Z0 = torch.tensor(atoms_to_Z(atoms), dtype=torch.long, device=device)

        if x0.ndim != 2 or x0.size(1) != 3:
            drop["coords_not_Nx3"] += 1
            continue

        if x0.size(0) != Z0.size(0):
            drop["atom_count_mismatch_atoms_vs_coords"] += 1
            continue

        # build bond graph with RemoveHs
        try:
            edges0, edge_attr0, n_atoms_mol = build_bond_graph_rdkit(smiles, device=device)
        except Exception:
            drop["rdkit_parse_or_bond_fail"] += 1
            continue

        if n_atoms_mol != x0.size(0):
            drop["atom_count_mismatch_rdkit_vs_coords"] += 1
            continue

        # if no bonds -> fallback radius
        if edges0.size(1) == 0:
            if drop_if_no_bonds:
                drop["no_bonds_dropped"] += 1
                continue
            # radius fallback
            # edges0 = build_radius_graph_torch(x0, r_cut=r_cut)  # (2,E)
            # if edges0.size(1) == 0:
            #     drop["radius_no_edges"] += 1
            #     continue
            # edge_attr0 = radius_edge_attr_13(x0, edges0)

        # augment coords
        x1 = augment_coords(x0, rotate=rotate, noise_std=noise_std)
        x2 = augment_coords(x0, rotate=rotate, noise_std=noise_std)

        # edge dropout
        e1, ea1 = drop_edges(edges0, edge_attr0, drop_p=edge_drop)
        e2, ea2 = drop_edges(edges0, edge_attr0, drop_p=edge_drop)

        if e1.size(1) > 0: e1 = e1 + node_offset
        if e2.size(1) > 0: e2 = e2 + node_offset

        n = x0.size(0)
        Z1_list.append(Z0); x1_list.append(x1)
        b1_list.append(torch.full((n,), mol_idx, dtype=torch.long, device=device))
        e1_list.append(e1); ea1_list.append(ea1)

        Z2_list.append(Z0); x2_list.append(x2)
        b2_list.append(torch.full((n,), mol_idx, dtype=torch.long, device=device))
        e2_list.append(e2); ea2_list.append(ea2)

        ids.append(entry.get("idx", mol_idx))

        node_offset += n
        mol_idx += 1

    if mol_idx == 0:
        raise RuntimeError(f"No valid molecules in this batch after filtering. drop_stats={dict(drop)}")

    Z1 = torch.cat(Z1_list, dim=0)
    x1 = torch.cat(x1_list, dim=0)
    b1 = torch.cat(b1_list, dim=0)
    e1 = torch.cat(e1_list, dim=1) if len(e1_list) else torch.zeros((2,0), dtype=torch.long, device=device)
    ea1 = torch.cat(ea1_list, dim=0) if len(ea1_list) else torch.zeros((0,13), dtype=torch.float32, device=device)

    Z2 = torch.cat(Z2_list, dim=0)
    x2 = torch.cat(x2_list, dim=0)
    b2 = torch.cat(b2_list, dim=0)
    e2 = torch.cat(e2_list, dim=1) if len(e2_list) else torch.zeros((2,0), dtype=torch.long, device=device)
    ea2 = torch.cat(ea2_list, dim=0) if len(ea2_list) else torch.zeros((0,13), dtype=torch.float32, device=device)

    return (Z1, x1, e1, ea1, b1), (Z2, x2, e2, ea2, b2), ids
