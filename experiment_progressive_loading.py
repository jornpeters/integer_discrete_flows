# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import time
import torch
import torch.utils.data
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import os
from optimization.training import train, evaluate
from utils.load_data import load_dataset
from utils.plotting import plot_training_curve
import imageio


parser = argparse.ArgumentParser(description='PyTorch Discrete Normalizing flows')

parser.add_argument('-d', '--dataset', type=str, default='cifar10', choices=['cifar10', 'imagenet32', 'imagenet64'],
                    metavar='DATASET',
                    help='Dataset choice.')

parser.add_argument('-bs', '--batch_size', type=int, default=256, metavar='BATCH_SIZE',
                    help='input batch size for training (default: 100)')

parser.add_argument('--data_augmentation_level', type=int, default=2,
                    help='data augmentation level')

parser.add_argument('-nc', '--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}


def run(args, kwargs):

    args.snap_dir = snap_dir = \
        'snapshots/discrete_logisticcifar10_flows_2_levels_3__2019-09-27_13_08_49/'

    # ==================================================================================================================
    # SNAPSHOTS
    # ==================================================================================================================

    # ==================================================================================================================
    # LOAD DATA
    # ==================================================================================================================
    train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)

    final_model = torch.load(snap_dir + 'a.model')
    if hasattr(final_model, 'module'):
        final_model = final_model.module

    from models.backround import SmoothRound
    for module in final_model.modules():
        if isinstance(module, SmoothRound):
            module._round_decay = 1.

    exp_dir = snap_dir + 'partials/'
    os.makedirs(exp_dir, exist_ok=True)

    images = []
    with torch.no_grad():
        for data, _ in test_loader:

            if args.cuda:
                data = data.cuda()

            for i in range(len(data)):
                _, _, _, pz, z, pys, ys, ldj = final_model.forward(data[i:i+1])

                for j in range(len(ys) + 1):
                    x_recon = final_model.inverse(
                        z,
                        ys[len(ys) - j:])

                    images.append(x_recon.float())

                if i == 10:
                    break
            break

    for j in range(len(ys) + 1):

        grid = make_grid(
            torch.stack(images[j::len(ys) + 1], dim=0).squeeze(),
            nrow=11, padding=0,
            normalize=True, range=None,
            scale_each=False, pad_value=0)

        imageio.imwrite(
            exp_dir + 'loaded{j}.png'.format(j=j),
            grid.cpu().numpy().transpose(1, 2, 0))


if __name__ == "__main__":

    run(args, kwargs)
