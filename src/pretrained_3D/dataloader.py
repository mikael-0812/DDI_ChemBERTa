from rdkit import Chem
pt = Chem.GetPeriodicTable()

import torch
from torch.utils.data import Dataset

class PT3DDataset(Dataset):
    """
    pt_obj[k] = {'smiles': str, 'atoms': list[str], 'confs': list[np.ndarray (3,)]}
    """
    def __init__(self, pt_path: str):
        self.obj = torch.load(pt_path, map_location="cpu")
        self.keys = list(self.obj.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        e = self.obj[k]
        return int(k), e["smiles"], e["atoms"], e["confs"]

