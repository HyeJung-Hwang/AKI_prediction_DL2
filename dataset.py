from torch.utils.data import Dataset
from torch import Tensor
import torch

class ECGDataset(Dataset):
    def __init__(self,ecg_data ):
        self.x_data = ecg_data
    def __len__(self):
        return len(self.x_data)
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.x_data[idx])
        return x, y