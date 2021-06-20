import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.data.data_loader import DigitDataset
from src.model.classification_model import Net


def train(tr_data, te_data=None):
    # gpu 설정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('학습을 진행하는 기기:', device)
    print('gpu 개수:', torch.cuda.device_count())

    # model
    model = Net()
    model.to(device)

    # loss function, optimizer 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # train
    for epoch in range(30):
        train_loss = 0.0
        for i, data in enumerate(tr_data):
            img, lbl = data
            img = img.float().to(device)
            lbl = lbl.to(device)

            optimizer.zero_grad()

            tr_outputs = model(img)
            tr_loss = criterion(tr_outputs, lbl)
            tr_loss.backward()
            optimizer.step()

            train_loss += tr_loss.item()
            print("\r>>>>>>>>>>>>>>>>>>>mini batch {} loss: {}".format(i, tr_loss), end='')

        train_loss /= len(tr_data_loader)
        print("\r{} epoch train loss: {}".format(epoch, train_loss))

    torch.cuda.empty_cache()


if __name__ == "__main__":
    img_trsf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])]  # (input[channel] - mean[channel]) / std[channel]
    )

    train_dataset = DigitDataset("../data/train.csv", img_trsf=img_trsf)
    tr_data_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)

    # test_dataset = DigitDataset("../data/test.csv", img_trsf=img_trsf)
    # te_data_loader = DataLoader(test_dataset, batch_size=2)

    train(tr_data_loader)

