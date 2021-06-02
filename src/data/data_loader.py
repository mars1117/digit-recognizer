import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader


class DigitDataset(Dataset):
    def __init__(self, scv_file, root_dir, transform=None):
        self.digit_data = pd.read_csv(scv_file)

    def __len__(self):
        return len(self.digit_data)

    def __getitem__(self, item):
        


if __name__ == "__main__":
    print("data loader module")
    print(os.getcwd())

    train_data = pd.read_csv("../../data/train.csv")