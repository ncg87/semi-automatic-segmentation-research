import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    # -- Initializes dataset
    def __init__(self, x, y):
        self.x = x
        self.y = y
    # -- Returns image and mask at a certain index
    def __getitem__(self, idx):
        img = self.x[idx]
        mask = self.y[idx]
        return img, mask
    # -- Returns length of dataset
    def __len__(self):
        return len(self.x)