import json
from random import random

import numpy as np
from rdkit import Chem

from utils import is_all_zeros, is_z_all_zeros

pt = Chem.GetPeriodicTable()

import torch
from torch.utils.data import Dataset

class PT3DDataset(Dataset):
    """
    Input .pt format:
        obj[id] = {"smiles" : smiles, "atoms" : atoms, "confs" : np.ndarray (N, 3)}
        Save 3D conformers and fallback 2D graph (with z = all_zeros)
        '../data/train.pt' : 1348 drugs
    """
    def __init__(
        self,
        pt_path: str,
        remove_multifragment: bool = False, #salts/ions multi-fragments
        min_atoms: int = 3, #removes drug molecules including only salts and metal ions
        max_atoms: int = 256,
        drop_all_zeros: bool = True,
        drop_z_all_zeros: bool = False,
        seed: int = 33,
        save_valid_ids: str = None,
    ):
        super().__init__()
        # random.seed(seed)
        # np.random.seed(seed)

        self.obj = torch.load(pt_path, map_location='cpu')
        self.keys_all = sorted([int(k) for k in self.obj.keys()])

        self.valid_keys = []
        self.invalid = {} #id -> reason

        for k in self.keys_all:
            entry = self.obj[k]
            smiles = entry.get("smiles", "")
            atoms = entry.get("atoms", None)
            conformers = entry.get("confs", None)

            if atoms is None or conformers is None:
                self.invalid[k] = 'missing'
                continue

            coords = np.asarray(conformers, dtype=np.float32)
            if coords.ndim != 2 or coords.shape[1] != 3:
                self.invalid[k] = 'invalid_coords'
                continue

            n = coords.shape[0]
            if n < min_atoms:
                self.invalid[k] = 'min_atoms_exception'
                continue

            if n > max_atoms:
                self.invalid[k] = 'max_atoms_exception'
                smiles = smiles[:max_atoms]

            if remove_multifragment and isinstance(smiles, str) and ('.' in smiles):
                self.invalid[k] = 'remove_multifragment_exception'
                continue

            if drop_all_zeros and is_all_zeros(coords):
                self.invalid[k] = 'drop_all_zeros_exception'
                continue

            if drop_z_all_zeros and is_z_all_zeros(coords):
                self.invalid[k] = 'drop_z_all_zeros_exception'
                continue

            self.valid_keys.append(k)

        if save_valid_ids is not None:
            with open(save_valid_ids, 'w') as f:
                json.dump({"valid_keys": self.valid_keys, "invalid": self.invalid}, f, indent=2)


        print(f"[Dataset] total: {len(self.valid_keys)}")
        print(f"[Dataset] valid: {len(self.valid_keys)}")
        print(f"[Dataset] invalid: {len(self.invalid)}")

    def __len__(self):
        return len(self.valid_keys)

    def __getitem__(self, idx):
        k = self.valid_keys[idx]
        entry = self.obj[k]
        atoms = entry.get("atoms", None)
        coords = np.asarray(entry["confs"], dtype=np.float32)
        smiles = entry.get("smiles", "")

        return {
            "idx": int(k),
            "smiles": smiles,
            "atoms": atoms,
            "confs": coords,
        }


