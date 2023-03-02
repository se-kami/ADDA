#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import torch
from tqdm import tqdm
from torch import nn
from adda.model import CNN, Discriminator
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


def domain_adapt(config, to_cli=True):
    if to_cli:
        print('Starting domain adaptation')

    # data
    loader_train_src = config['loader']['train_src']
    loader_dev_src = config['loader']['dev_src']
    loader_train_trg = config['loader']['train_trg']
    loader_dev_trg = config['loader']['dev_trg']
    # models
    cnn_src = config['model']['cnn_src']
    cnn_trg = config['model']['cnn_trg']
    discriminator = config['model']['discriminator']
    # device
    device = config['device']
    cnn_src = cnn_src.to(device).eval()
    cnn_trg = cnn_trg.to(device)
    discriminator = discriminator.to(device)
    # loss function
    loss_fn = config['loss_fn']
    # optimizer
    optimizer_cnn = config['optimizer']['optimizer_cnn']
    optimizer_kwargs_cnn = config['optimizer']['kwargs_cnn']
    optimizer_disc = config['optimizer']['optimizer_disc']
    optimizer_kwargs_disc = config['optimizer']['kwargs_disc']
    optimizer_cnn = optimizer_cnn(cnn_trg.parameters(), **optimizer_kwargs_cnn)
    optimizer_disc = optimizer_disc(discriminator.parameters(), **optimizer_kwargs_disc)
    # training
    epochs = config['epochs']

    cnn_trg_best = None
    discriminator_best = None
    loss_best = None
    bar_epoch = tqdm(range(epochs))
    for epoch in bar_epoch:
        # training
        cnn_trg.train()
        discriminator.train()
        loader_train = zip(loader_train_src, loader_train_trg)
        bar_train = tqdm(loader_train, leave=False)
        correct_epoch_src = 0.0
        correct_epoch_trg = 0.0
        total_epoch_src = 0.0
        total_epoch_trg = 0.0
        loss_epoch_src = 0.0
        loss_epoch_trg = 0.0
        for (x_src, _), (x_trg, _) in bar_train:
            x_src = x_src.to(device)
            x_trg = x_trg.to(device)

            # train discriminator
            x_emb_src = cnn_src(x_src)
            x_emb_trg = cnn_trg(x_trg)

            y_src_pred = discriminator(x_emb_src).view(-1)
            y_trg_pred = discriminator(x_emb_trg).view(-1)

            y_src = torch.ones(y_src_pred.shape[0], device=device).float()
            y_trg = torch.zeros(y_trg_pred.shape[0], device=device).float()

            loss_src = loss_fn(y_src_pred, y_src)
            loss_trg = loss_fn(y_trg_pred, y_trg)

            loss_disc = 0.5 * loss_src + 0.5 * loss_trg

            optimizer_disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()

            # train target cnn
            x_emb_trg = cnn_trg(x_trg)
            y_trg_pred = discriminator(x_emb_trg).view(-1)
            y_trg = torch.ones(y_trg_pred.shape[0], device=device).float()
            loss_trg = loss_fn(y_trg_pred, y_trg)
            loss_cnn = loss_trg
            optimizer_cnn.zero_grad()
            loss_cnn.backward()
            optimizer_cnn.step()

            # logging
            correct_epoch_src += (y_src_pred <= 0.0).float().sum()
            correct_epoch_trg += (y_trg_pred > 0.0).float().sum()
            total_epoch_src += len(x_emb_src)
            total_epoch_trg += len(x_emb_trg)
            loss_epoch_src += loss_src.item() * len(x_emb_src)
            loss_epoch_trg += loss_trg.item() * len(x_emb_trg)

            postfix_train = {}
            postfix_train['acc_train_src'] = f"{100*correct_epoch_src/total_epoch_src:02.1f}"
            postfix_train['acc_train_trg'] = f"{100*correct_epoch_trg/total_epoch_trg:02.1f}"
            postfix_train['loss_train_src'] = f"{loss_epoch_src/total_epoch_src:02.2f}"
            postfix_train['loss_train_trg'] = f"{loss_epoch_trg/total_epoch_trg:02.2f}"
            bar_train.set_postfix(postfix_train)

        # validation
        cnn_trg.eval()
        discriminator.eval()
        # validation src
        bar_dev_src = tqdm(loader_dev_src, leave=False)
        correct_epoch_src = 0.0
        total_epoch_src = 0.0
        loss_epoch_src = 0.0
        with torch.no_grad():
            for x, _ in bar_dev_src:
                x = x.to(device)
                x_emb = cnn_src(x)
                y_pred = discriminator(x_emb).view(-1)
                y_src = torch.ones(y_pred.shape[0], device=device).float()
                loss = loss_fn(y_pred, y_src)

                # logging
                correct_epoch_src += (y_pred <= 0.0).float().sum().item()
                total_epoch_src += len(x)
                loss_epoch_src += loss.item() * len(x)

                postfix_dev = {}
                postfix_dev['acc_dev_src'] = f"{100*correct_epoch_src/total_epoch_src:02.1f}"
                postfix_dev['loss_dev_src'] = f"{loss_epoch_src/total_epoch_src:02.2f}"
                bar_dev_src.set_postfix(postfix_dev)

        # validation trg
        bar_dev_trg = tqdm(loader_dev_trg, leave=False)
        correct_epoch_trg = 0.0
        total_epoch_trg = 0.0
        loss_epoch_trg = 0.0
        with torch.no_grad():
            for x, _ in bar_dev_trg:
                x = x.to(device)
                x_emb = cnn_trg(x)
                y_pred = discriminator(x_emb).view(-1)
                y_trg = torch.zeros(y_pred.shape[0], device=device).float()
                loss = loss_fn(y_pred, y_trg)

                # logging
                correct_epoch_trg += (y_pred > 0.0).float().sum().item()
                total_epoch_trg += len(x)
                loss_epoch_trg += loss.item() * len(x)

                postfix_dev['acc_dev_trg'] = f"{100*correct_epoch_trg/total_epoch_trg:02.1f}"
                postfix_dev['loss_dev_trg'] = f"{loss_epoch_trg/total_epoch_trg:02.2f}"
                bar_dev_trg.set_postfix(postfix_dev)


        # keep best model
        loss = loss_epoch_src + loss_epoch_trg
        if loss_best is None or loss < loss_best:
            cnn_trg_best = copy.deepcopy(cnn_trg).cpu()
            discriminator_best = copy.deepcopy(discriminator).cpu()
            loss_best = loss

        bar_epoch.set_postfix(
                dict(**postfix_train, **postfix_dev, loss_best=loss_best))

    return cnn_trg_best, discriminator_best


if __name__ == '__main__':
    # data
    data_dir = '_DATA'
    cnn_src_path = 'cnn_src.pt'
    seed = 1234
    from adda.data import get_loaders_svhn, get_loaders_mnist

    # models
    size_mid = 500
    cnn_src = CNN(size_out=size_mid)
    cnn_trg = CNN(size_out=size_mid)
    discriminator = Discriminator(size_in=size_mid)
    cnn_src.load_state_dict(torch.load(cnn_src_path))
    cnn_trg.load_state_dict(torch.load(cnn_src_path))

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # domain adaptation config
    config_adapt = {}
    lr_adapt_cnn = 3e-4
    lr_adapt_disc = 3e-4
    epochs_adapt = 32
    batch_size_adapt = 1024
    optimizer_adapt_cnn = Adam
    optimizer_adapt_disc = Adam

    loader_train_src, loader_dev_src, _ = get_loaders_svhn(
            data_dir=data_dir,
            batch_size=batch_size_adapt,
            seed=seed,
            )
    loader_train_trg, loader_dev_trg, _ = get_loaders_mnist(
            data_dir=data_dir,
            batch_size=batch_size_adapt,
            seed=seed,
            )

    config_adapt['model'] = {}
    config_adapt['model']['cnn_src'] = cnn_src
    config_adapt['model']['cnn_trg'] = cnn_trg
    config_adapt['model']['discriminator'] = discriminator

    config_adapt['loader'] = {}
    config_adapt['loader']['train_src'] = loader_train_src
    config_adapt['loader']['dev_src'] = loader_dev_src
    config_adapt['loader']['train_trg'] = loader_train_trg
    config_adapt['loader']['dev_trg'] = loader_dev_trg

    config_adapt['loss_fn'] = nn.BCEWithLogitsLoss()

    config_adapt['optimizer'] = {}
    config_adapt['optimizer']['optimizer_cnn'] = optimizer_adapt_cnn
    config_adapt['optimizer']['kwargs_cnn'] = {'lr': lr_adapt_cnn,
                                               'betas': (0.5, 0.9)}
    config_adapt['optimizer']['optimizer_disc'] = optimizer_adapt_disc
    config_adapt['optimizer']['kwargs_disc'] = {'lr': lr_adapt_disc,
                                                'betas': (0.5, 0.9)}

    config_adapt['epochs'] = epochs_adapt
    config_adapt['device'] = device

    cnn_trg, discriminator = domain_adapt(config_adapt)
    torch.save(cnn_trg.cpu().state_dict(), 'cnn_trg.pt')
    torch.save(discriminator.cpu().state_dict(), 'discriminator.pt')
