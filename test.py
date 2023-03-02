#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import torch
from tqdm import tqdm
from torch import nn
from adda.model import CNN, Classifier
from torch.optim import Adam

def check_accuracy(cnn, classifier, loader, device):
    total = 0
    correct = 0

    cnn.eval()
    classifier.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).view(-1)
            # forward pass
            embeddings = cnn(x)
            labels = classifer(embeddings)
            labels = torch.argmax(labels, axis=1).view(-1)
            # add totals
            total += len(x)
            correct += torch.sum(labels == y).item()

    return correct / total


def test(config, to_cli=True):
    if to_cli:
        print('Starting testing')

    # data
    loader_train = config['loader']['train']
    loader_dev = config['loader']['dev']
    # models
    cnn = config['model']['cnn']
    classifier = config['model']['classifier']
    # device
    device = config['device']
    cnn = cnn.to(device)
    classifier = classifier.to(device)


    with torch.no_grad():
        # train set
        correct_epoch_train = 0.0
        total_epoch_train = 0.0
        bar_train = tqdm(loader_train, leave=False)
        for x, y in bar_train:
            x = x.to(device)
            y = y.to(device).view(-1)

            x_emb = cnn(x)
            y_pred = classifier(x_emb)

            # logging
            correct_epoch_train += (torch.argmax(y_pred, 1).view(-1) == y).float().sum()
            total_epoch_train += len(x)

            postfix_train = {}
            postfix_train['acc_train'] = f"{100*correct_epoch_train/total_epoch_train:02.1f}"
            bar_train.set_postfix(postfix_train)


        # dev set
        correct_epoch_dev = 0.0
        total_epoch_dev = 0.0
        bar_dev = tqdm(loader_dev, leave=False)
        for x, y in bar_dev:
            x = x.to(device)
            y = y.to(device).view(-1)

            x_emb = cnn(x)
            y_pred = classifier(x_emb)

            # logging
            correct_epoch_dev += (torch.argmax(y_pred, 1).view(-1) == y).float().sum()
            total_epoch_dev += len(x)

            postfix_dev = {}
            postfix_dev['acc_dev'] = f"{100*correct_epoch_dev/total_epoch_dev:02.1f}"
            bar_dev.set_postfix(postfix_dev)

    acc_train = 100 * correct_epoch_train / total_epoch_train
    acc_dev = 100 * correct_epoch_dev / total_epoch_dev
    return acc_train, acc_dev


if __name__ == '__main__':
    # data
    data_dir = '_DATA'
    cnn_trg_path = 'cnn_trg.pt'
    classifier_path = 'classifier.pt'
    from adda.data import get_loaders_svhn, get_loaders_mnist
    seed = 1234

    # models
    size_mid=500
    cnn_trg = CNN(size_out=size_mid)
    classifier = Classifier(size_in=size_mid, size_out=10)
    cnn_trg.load_state_dict(torch.load(cnn_trg_path))
    classifier.load_state_dict(torch.load(classifier_path))

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size_test = 1024

    loader_train_trg, loader_dev_trg, _ = get_loaders_mnist(
            data_dir=data_dir,
            batch_size=batch_size_test,
            seed=seed,
            )

    # testing config
    config_test = {}
    config_test['loader'] = {}
    config_test['loader']['train'] = loader_train_trg
    config_test['loader']['dev'] = loader_dev_trg

    config_test['model'] = {}
    config_test['model']['cnn'] = cnn_trg
    config_test['model']['classifier'] = classifier

    config_test['device'] = device
    acc_train, acc_dev = test(config_test)

    print(f"Dev set accuracy {acc_dev}")
