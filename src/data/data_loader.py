import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DigitDataset(Dataset):
    def __init__(self, scv_file, img_trsf=None, mode='train'):
        self.digit_data = pd.read_csv(scv_file)
        self.img_square_len = 28

        if mode == "train":
            self.data = self.digit_data.iloc[:, 1:]
            self.label = self.digit_data["label"]
        elif mode == "eval":
            self.data = self.digit_data
            self.label = None

        self.image_transform = img_trsf

    def __len__(self):
        return len(self.digit_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data.iloc[idx, :]
        reshape_image = np.reshape(image.to_numpy(dtype='float64'), (self.img_square_len, self.img_square_len))
        if self.label is not None:
            label = torch.tensor(self.label[idx])
        else:
            label = torch.tensor(0)

        if self.image_transform:
            reshape_image = self.image_transform(reshape_image)
            reshape_image /= reshape_image.max()

        return reshape_image, label


if __name__ == "__main__":
    img_trsf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])]   # (input[channel] - mean[channel]) / std[channel]
    )

    digit_dataset = DigitDataset("../../data/train.csv", img_trsf=img_trsf, mode="train")

    for i in range(1):
        img, lbl = digit_dataset[i]
        print(type(lbl))
        plt.imshow(img.numpy().squeeze(), cmap='gray')
        plt.show()
