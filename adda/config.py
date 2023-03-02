#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.optim import SGD
from .data import get_mnist_data, get_svhn_data
from .data import get_loaders
from .model import DANN_SVHN

class Config():
    """
    Default Config class
    """
    # data
    DATA_DIR = '_DATA'

    # LAMBDA
    GAMMA = 10

    # OPTIMIZER
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    ALPHA = 10
    BETA = 0.75

    def get_optimizer(self):
        return self.optimizer

    def get_model(self):
        return self.model

    def get_iters(self):
        return self.n_iters

    def get_log_settings(self):
        return self.log_every

    def get_device(self):
        return self.device


class ConfigSvhnMnist(Config):
    """
    Config for SVHN -> MNIST experiment
    """
    def __init__(self, n_iters=10000, batch_size=32, seed=None, log_every=500):
        """
        args:
            n_iters: int
                number of training iterations
            batch_size: int
                batch size
            seed: int
                random seed for train-dev split
            log_every: int
                how often to perform evalution on dev set
        """
        # data
        self.seed = seed
        self.batch_size = batch_size

        # training loop
        self.n_iters = n_iters
        self.log_every = log_every

        # model
        self.model = DANN_SVHN()

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.LEARNING_RATE, momentum=self.MOMENTUM)

        # device

