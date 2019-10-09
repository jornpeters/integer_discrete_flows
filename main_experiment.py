# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import time
import torch
import torch.utils.data
import torch.optim as optim
import numpy as np
import math
import random

import os

import datetime

from optimization.training import train, evaluate
from utils.load_data import load_dataset

parser = argparse.ArgumentParser(description='PyTorch Discrete Normalizing flows')

parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'imagenet32', 'imagenet64'],
                    metavar='DATASET',
                    help='Dataset choice.')

parser.add_argument('-nc', '--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--manual_seed', type=int, help='manual seed, if not given resorts to random seed.')

parser.add_argument('-li', '--log_interval', type=int, default=20, metavar='LOG_INTERVAL',
                    help='how many batches to wait before logging training status')

parser.add_argument('--evaluate_interval_epochs', type=int, default=25,
                    help='Evaluate per how many epochs')

parser.add_argument('-od', '--out_dir', type=str, default='snapshots', metavar='OUT_DIR',
                    help='output directory for model snapshots etc.')

fp = parser.add_mutually_exclusive_group(required=False)
fp.add_argument('-te', '--testing', action='store_true', dest='testing',
                help='evaluate on test set after training')
fp.add_argument('-va', '--validation', action='store_false', dest='testing',
                help='only evaluate on validation set')
parser.set_defaults(testing=True)

# optimization settings
parser.add_argument('-e', '--epochs', type=int, default=2000, metavar='EPOCHS',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('-es', '--early_stopping_epochs', type=int, default=300, metavar='EARLY_STOPPING',
                    help='number of early stopping epochs')

parser.add_argument('-bs', '--batch_size', type=int, default=256, metavar='BATCH_SIZE',
                    help='input batch size for training (default: 100)')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, metavar='LEARNING_RATE',
                    help='learning rate')
parser.add_argument('--warmup', type=int, default=10,
                    help='number of warmup epochs')

parser.add_argument('--data_augmentation_level', type=int, default=2,
                    help='data augmentation level')

parser.add_argument('--variable_type', type=str, default='discrete',
                    help='variable type of data distribution: discrete/continuous',
                    choices=['discrete', 'continuous'])
parser.add_argument('--distribution_type', type=str, default='logistic',
                    choices=['logistic', 'normal', 'steplogistic'],
                    help='distribution type: logistic/normal')
parser.add_argument('--n_flows', type=int, default=8,
                    help='number of flows per level')
parser.add_argument('--n_levels', type=int, default=3,
                    help='number of levels')

parser.add_argument('--n_bits', type=int, default=8,
                    help='')

# ---------------- SETTINGS CONCERNING NETWORKS -------------
parser.add_argument('--densenet_depth', type=int, default=8,
                    help='Depth of densenets')
parser.add_argument('--n_channels', type=int, default=512,
                    help='number of channels in coupling and splitprior')
# ---------------- ----------------------------- -------------


# ---------------- SETTINGS CONCERNING COUPLING LAYERS -------------
parser.add_argument('--coupling_type', type=str, default='shallow',
                    choices=['shallow', 'resnet', 'densenet'],
                    help='Type of coupling layer')
parser.add_argument('--splitfactor', default=0, type=int,
                    help='Split factor for coupling layers.')

parser.add_argument('--split_quarter', dest='split_quarter', action='store_true',
                    help='Split coupling layer on quarter')
parser.add_argument('--no_split_quarter', dest='split_quarter', action='store_false')
parser.set_defaults(split_quarter=True)
# ---------------- ----------------------------------- -------------


# ---------------- SETTINGS CONCERNING SPLITPRIORS -------------
parser.add_argument('--splitprior_type', type=str, default='shallow',
                    choices=['none', 'shallow', 'resnet', 'densenet'],
                    help='Type of splitprior. Use \'none\' for no splitprior')
# ---------------- ------------------------------- -------------


# ---------------- SETTINGS CONCERNING PRIORS -------------
parser.add_argument('--n_mixtures', type=int, default=1,
                    help='number of mixtures')
# ---------------- ------------------------------- -------------

parser.add_argument('--hard_round', dest='hard_round', action='store_true',
                    help='Rounding of translation in discrete models. Weird '
                    'probabilistic implications, only for experimental phase')
parser.add_argument('--no_hard_round', dest='hard_round', action='store_false')
parser.set_defaults(hard_round=True)

parser.add_argument('--round_approx', type=str, default='smooth',
                    choices=['smooth', 'stochastic'])

parser.add_argument('--lr_decay', default=0.999, type=float,
                    help='Learning rate')

parser.add_argument('--temperature', default=1.0, type=float,
                    help='Temperature used for BackRound. It is used in '
                    'the the SmoothRound module. '
                    '(default=1.0')

# gpu/cpu
parser.add_argument('--gpu_num', type=int, default=0, metavar='GPU',
                    help='choose GPU to run on.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.manual_seed is None:
    args.manual_seed = random.randint(1, 100000)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
np.random.seed(args.manual_seed)


kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}


def run(args, kwargs):

    print('\nMODEL SETTINGS: \n', args, '\n')
    print("Random Seed: ", args.manual_seed)

    if 'imagenet' in args.dataset and args.evaluate_interval_epochs > 5:
        args.evaluate_interval_epochs = 5

    # ==================================================================================================================
    # SNAPSHOTS
    # ==================================================================================================================
    args.model_signature = str(datetime.datetime.now())[0:19].replace(' ', '_')
    args.model_signature = args.model_signature.replace(':', '_')

    snapshots_path = os.path.join(args.out_dir, args.variable_type + '_' + args.distribution_type + args.dataset)
    snap_dir = snapshots_path

    snap_dir += '_' + 'flows_' + str(args.n_flows) + '_levels_' + str(args.n_levels)

    snap_dir = snap_dir + '__' + args.model_signature + '/'

    args.snap_dir = snap_dir

    if not os.path.exists(snap_dir):
        os.makedirs(snap_dir)

    with open(snap_dir + 'log.txt', 'a') as ff:
        print('\nMODEL SETTINGS: \n', args, '\n', file=ff)

    # SAVING
    torch.save(args, snap_dir + '.config')

    # ==================================================================================================================
    # LOAD DATA
    # ==================================================================================================================
    train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)

    # ==================================================================================================================
    # SELECT MODEL
    # ==================================================================================================================
    # flow parameters and architecture choice are passed on to model through args
    print(args.input_size)

    import models.Model as Model

    model = Model.Model(args)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.set_temperature(args.temperature)
    model.enable_hard_round(args.hard_round)

    model_sample = model

    # ====================================
    # INIT
    # ====================================
    # data dependend initialization on CPU
    for batch_idx, (data, _) in enumerate(train_loader):
        model(data)
        break

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, dim=0)

    model.to(args.device)

    def lr_lambda(epoch):
        return min(1., (epoch+1) / args.warmup) * np.power(args.lr_decay, epoch)
    optimizer = optim.Adamax(model.parameters(), lr=args.learning_rate, eps=1.e-7)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    # ==================================================================================================================
    # TRAINING
    # ==================================================================================================================
    train_bpd = []
    val_bpd = []

    # for early stopping
    best_val_bpd = np.inf
    best_train_bpd = np.inf
    epoch = 0

    train_times = []

    model.eval()
    model.train()

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        scheduler.step()
        tr_loss, tr_bpd = train(epoch, train_loader, model, optimizer, args)
        train_bpd.append(tr_bpd)
        train_times.append(time.time()-t_start)
        print('One training epoch took %.2f seconds' % (time.time()-t_start))

        if epoch < 25 or epoch % args.evaluate_interval_epochs == 0:
            v_loss, v_bpd = evaluate(
                train_loader, val_loader, model, model_sample, args,
                epoch=epoch, file=snap_dir + 'log.txt')

            val_bpd.append(v_bpd)

            # Model save based on TRAIN performance (is heavily correlated with validation performance.)
            if np.mean(tr_bpd) < best_train_bpd:
                best_train_bpd = np.mean(tr_bpd)
                best_val_bpd = v_bpd
                torch.save(model.module, snap_dir + 'a.model')
                torch.save(optimizer, snap_dir + 'a.optimizer')
                print('->model saved<-')

            print('(BEST: train bpd {:.4f}, test bpd {:.4f})\n'.format(
                best_train_bpd, best_val_bpd))

            if math.isnan(v_loss):
                raise ValueError('NaN encountered!')

    train_bpd = np.hstack(train_bpd)
    val_bpd = np.array(val_bpd)

    # training time per epoch
    train_times = np.array(train_times)
    mean_train_time = np.mean(train_times)
    std_train_time = np.std(train_times, ddof=1)
    print('Average train time per epoch: %.2f +/- %.2f' % (mean_train_time, std_train_time))

    # ==================================================================================================================
    # EVALUATION
    # ==================================================================================================================
    final_model = torch.load(snap_dir + 'a.model')
    test_loss, test_bpd = evaluate(
        train_loader, test_loader, final_model, final_model, args,
        epoch=epoch, file=snap_dir + 'test_log.txt')

    print('Test loss / bpd: %.2f / %.2f' % (test_loss, test_bpd))


if __name__ == "__main__":

    run(args, kwargs)
