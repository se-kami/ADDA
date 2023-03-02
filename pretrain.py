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


def pretrain(config, to_cli=True):
    if to_cli:
        print('Starting pretraining')

    # data
    loader_train = config['loader']['train']
    loader_dev = config['loader']['dev']
    # models
    cnn = config['model']['cnn']
    classifier = config['model']['classifier']
    all_params = list(cnn.parameters()) + list(classifier.parameters())
    # device
    device = config['device']
    cnn = cnn.to(device)
    classifier = classifier.to(device)
    # loss function
    loss_fn = config['loss_fn']
    # optimizer
    optimizer = config['optimizer']['optimizer']
    optimizer_kwargs = config['optimizer']['kwargs']
    optimizer = optimizer(all_params, **optimizer_kwargs)
    # training
    epochs = config['epochs']


    cnn_best = None
    classifier_best = None
    acc_best = 0.0
    bar_epoch = tqdm(range(epochs))
    for epoch in bar_epoch:
        # training
        cnn.train()
        classifier.train()
        bar_train = tqdm(loader_train, leave=False)
        correct_epoch = 0.0
        total_epoch = 0.0
        loss_epoch = 0.0
        for x, y in bar_train:
            x = x.to(device)
            y = y.to(device)

            x_emb = cnn(x)
            y_pred = classifier(x_emb)

            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            correct_epoch += (torch.argmax(y_pred, 1).view(-1) == y).float().sum()
            total_epoch += len(x)
            loss_epoch += loss.item() * len(x)

            postfix_train = {}
            postfix_train['acc_train'] = f"{100*correct_epoch/total_epoch:02.1f}"
            postfix_train['loss_train'] = f"{loss_epoch/total_epoch:02.2f}"
            bar_train.set_postfix(postfix_train)

        # validation
        cnn.eval()
        classifier.eval()
        bar_dev = tqdm(loader_dev, leave=False)
        correct_epoch = 0.0
        total_epoch = 0.0
        loss_epoch = 0.0
        with torch.no_grad():
            for x, y in bar_dev:
                x = x.to(device)
                y = y.to(device)

                x_emb = cnn(x)
                y_pred = classifier(x_emb)

                loss = loss_fn(y_pred, y)

                # logging
                correct_epoch += (torch.argmax(y_pred, 1).view(-1) == y).float().sum().item()
                total_epoch += len(x)
                loss_epoch += loss.item() * len(x)

                postfix_dev = {}
                postfix_dev['acc_dev'] = f"{100*correct_epoch/total_epoch:02.1f}"
                postfix_dev['loss_dev'] = f"{loss_epoch/total_epoch:02.2f}"
                bar_dev.set_postfix(postfix_dev)

        # keep best model
        accuracy = correct_epoch / total_epoch
        if 100 * accuracy > acc_best:
            cnn_best = copy.deepcopy(cnn).cpu()
            classifier_best = copy.deepcopy(classifier).cpu()
            acc_best = 100 * accuracy

        bar_epoch.set_postfix(
                dict(**postfix_train, **postfix_dev, acc_best=acc_best))

    return cnn_best, classifier_best


if __name__ == '__main__':
    # data
    data_dir = '_DATA'
    from adda.data import get_loaders_svhn
    seed = 1234

    # models
    size_mid = 500
    cnn_src = CNN(size_out=size_mid)
    classifier = Classifier(size_in=size_mid, size_out=10)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pretrain config
    lr_pretrain = 3e-4
    epochs_pretrain = 32
    batch_size_pretrain = 1024
    optimizer_pretrain = Adam

    loader_train, loader_dev, _ = get_loaders_svhn(
            data_dir=data_dir,
            batch_size=batch_size_pretrain,
            seed=seed,
            )

    config_pretrain = dict()

    config_pretrain['model'] = dict()
    config_pretrain['model']['cnn'] = cnn_src
    config_pretrain['model']['classifier'] = classifier

    config_pretrain['loader'] = dict()
    config_pretrain['loader']['train'] = loader_train
    config_pretrain['loader']['dev'] = loader_dev

    config_pretrain['loss_fn'] = nn.CrossEntropyLoss()

    config_pretrain['optimizer'] = dict()
    config_pretrain['optimizer']['optimizer'] = optimizer_pretrain
    config_pretrain['optimizer']['kwargs'] = {'lr': lr_pretrain}

    config_pretrain['epochs'] = epochs_pretrain
    config_pretrain['device'] = device

    cnn_best, classifier_best = pretrain(config_pretrain)
    torch.save(cnn_best.cpu().state_dict(), 'cnn_src.pt')
    torch.save(classifier_best.cpu().state_dict(), 'classifier.pt')
