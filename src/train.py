import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.data_loader import DigitDataset


def train():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print('학습을 진행하는 기기:', device)
    print('gpu 개수:', torch.cuda.device_count())


if __name__ == "__main__":
    img_trsf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])]  # (input[channel] - mean[channel]) / std[channel]
    )

    digit_dataset = DigitDataset("../data/train.csv", img_trsf=img_trsf)

    for i in range(1):
        img, lbl = digit_dataset[i]
        print(type(lbl))
        plt.imshow(img.numpy().squeeze(), cmap='gray')
        plt.show()

    train()
