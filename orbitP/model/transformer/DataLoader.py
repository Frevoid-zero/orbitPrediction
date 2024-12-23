import sys

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import joblib
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
class orbitPDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, axis, training_length, forecast_window):
        
        # load raw data file
        self.data = data
        self.axis = axis
        self.T = training_length
        self.S = forecast_window

    def __len__(self):
        # 可用窗口的总数
        return len(self.data) - self.T - self.S + 1

    # Will pull an index between 0 and __len__. 
    def __getitem__(self, idx):

        # scalar is fit only to the input, to avoid the scaled values "leaking" information about the target range.
        # scalar is fit only for humidity, as the timestamps are already scaled
        # scalar input/output of shape: [n_samples, n_features].
        idx_pre = torch.tensor(np.array([i for i in range(idx,idx+self.T)]))
        idx_suf = torch.tensor(np.array([i for i in range(idx+self.T,idx+self.T+self.S)]))
        orbitData_pre = torch.tensor(self.data[idx:idx+self.T,:,self.axis])
        orbitData_suf = torch.tensor(self.data[idx+self.T:idx+self.T+self.S,:,self.axis])
        # print(orbitData_pre.size()) # (288,7)
        # print(orbitData_suf.size()) # (1,7)
        # sys.exit(0)
        return idx_pre, idx_suf, orbitData_pre, orbitData_suf, self.T, self.S