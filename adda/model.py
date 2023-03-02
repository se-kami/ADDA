#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, size_out=500):
        super().__init__()

        # input: 3x32x32
        self.conv = nn.Sequential(
                nn.Conv2d(3, 20, 5),  # 20x28x28
                nn.ReLU(),
                nn.MaxPool2d(2),  # 6x14x14
                nn.Conv2d(20, 50, 5),  # 50x10x10
                nn.ReLU(),
                nn.MaxPool2d(2),  # 50x5x5
                nn.Flatten(),
                nn.Linear(50 * 5 * 5, size_out)
                )

    def forward(self, x):
        x = self.conv(x)
        return x


class Classifier(nn.Module):
    def __init__(self, size_in=500, size_out=10, size_mid=120):
        super().__init__()

        self.fc = nn.Sequential(
                nn.Linear(size_in, size_mid),
                nn.ReLU(),
                nn.Linear(size_mid, size_out),
                )

    def forward(self, x):
        x = self.fc(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, size_in, size_hidden=[500, 500]):
        super().__init__()

        net = []
        net.append(nn.Linear(size_in, size_hidden[0]))
        net.append(nn.ReLU())

        for i in range(len(size_hidden)-1):
            net.append(nn.Linear(size_hidden[i], size_hidden[i+1]))
            net.append(nn.ReLU())

        net.append(nn.Linear(size_hidden[-1], 1))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x
