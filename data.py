import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import glob
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from dataset import ECGDataset

def load_data(train_npy_path):

    train_set = ECGDataset(train_npy_path)
    valid_set = ECGDataset(train_npy_path)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=4, shuffle=True, num_workers=0)

    return train_loader, valid_loader


def load_test_data(test_npy_path, test_csv_path):
    test_df = pd.read_csv(test_csv_path, index_col=0)

    test_set = ECGDataset(test_npy_path, test_df)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

    return 