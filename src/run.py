import os
import numpy as np
import matplotlib.pyplot as plt
import csv

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from src.data.data_loader import DigitDataset
from src.model.classification_model import Net


def run(ev_data):
    # gpu 설정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('학습을 진행하는 기기:', device)
    print('gpu 개수:', torch.cuda.device_count())

    # checkpoint load
    checkpoint = torch.load('./checkpoints/digit_recog_0622.pth')

    # model
    model = Net()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    f = open('../data/submission.csv', 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(['ImageId', 'Label'])
    # evaluation
    for i, data in enumerate(ev_data):
        img, _ = data
        img = img.float().to(device)

        ev_outputs = model(img)
        result = np.argmax(ev_outputs.cpu().detach().numpy())
        wr.writerow([i+1, result])
        # print(ev_outputs)
        print(i+1, result)
        #
        # img = img.cpu().detach().numpy()
        # plt.imshow(np.squeeze(img))
        # plt.show()
    f.close()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    img_trsf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])]  # (input[channel] - mean[channel]) / std[channel]
    )

    eval_dataset = DigitDataset("../data/test.csv", img_trsf=img_trsf, mode="eval")
    ev_data_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    run(ev_data_loader)
    print('finish run model.')
