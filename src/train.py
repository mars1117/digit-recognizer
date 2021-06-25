import os
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

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
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # loss function, optimizer 설정
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # train
    minimum_tr_loss = 1.0
    for epoch in range(1000):
        train_loss = 0.0
        start_time = timer()
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

        train_loss /= len(tr_data)
        end_time = timer()
        print("\r{} epoch train loss: {:.4f}       elapsed time: {}".format(epoch, train_loss, (end_time - start_time)))

        if train_loss < minimum_tr_loss:
            minimum_tr_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': minimum_tr_loss,
            }, './checkpoints/digit_recog_0622.pth')

    torch.cuda.empty_cache()


if __name__ == "__main__":

    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')

    img_trsf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])]  # (input[channel] - mean[channel]) / std[channel]
    )

    train_dataset = DigitDataset("../data/train.csv", img_trsf=img_trsf, mode="train")
    tr_data_loader = DataLoader(train_dataset, batch_size=3600, shuffle=True, num_workers=0)

    train(tr_data_loader)
