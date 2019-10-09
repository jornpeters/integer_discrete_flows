from __future__ import print_function
import torch

from optimization.loss import calculate_loss
from utils.visual_evaluation import plot_reconstructions

import numpy as np


def train(epoch, train_loader, model, opt, args):
    model.train()
    train_loss = np.zeros(len(train_loader))
    train_bpd = np.zeros(len(train_loader))

    num_data = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, *args.input_size)

        data = data.to(args.device)

        opt.zero_grad()
        loss, bpd, bpd_per_prior, pz, z, pys, py, ldj = model(data)

        loss = torch.mean(loss)
        bpd = torch.mean(bpd)
        bpd_per_prior = [torch.mean(i) for i in bpd_per_prior]

        loss.backward()
        loss = loss.item()
        train_loss[batch_idx] = loss
        train_bpd[batch_idx] = bpd

        ldj = torch.mean(ldj).item() / np.prod(args.input_size) / np.log(2)

        opt.step()

        num_data += len(data)

        if batch_idx % args.log_interval == 0:
            perc = 100. * batch_idx / len(train_loader)

            tmp = 'Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)] \tLoss: {:11.6f}\tbpd: {:8.6f}\tbits ldj: {:8.6f}'
            print(tmp.format(epoch, num_data, len(train_loader.sampler), perc, loss, bpd, ldj))

            print('z min: {:8.3f}, max: {:8.3f}'.format(torch.min(z).item() * 256, torch.max(z).item() * 256))

            print('z  bpd: {:.3f}'.format(bpd_per_prior[0]))
            for i in range(1, len(bpd_per_prior)):
                print('y{} bpd: {:.3f}'.format(i-1, bpd_per_prior[i]))

            print('pz mu', np.mean(pz[0].data.cpu().numpy(), axis=(0, 1, 2, 3)))
            print('pz logs ', np.mean(pz[1].data.cpu().numpy(), axis=(0, 1, 2, 3)))
            if len(pz) == 3:
                print('pz pi   ', np.mean(pz[2].data.cpu().numpy(), axis=(0, 1, 2, 3)))

            for i, py in enumerate(pys):
                print('py{} mu   '.format(i), np.mean(py[0].data.cpu().numpy(), axis=(0, 1, 2, 3)))
                print('py{} logs '.format(i), np.mean(py[1].data.cpu().numpy(), axis=(0, 1, 2, 3)))

    from utils.visual_evaluation import plot_images
    import os
    if not os.path.exists(args.snap_dir + 'training/'):
        os.makedirs(args.snap_dir + 'training/')

    print('====> Epoch: {:3d} Average train loss: {:.4f}, average bpd: {:.4f}'.format(
        epoch, train_loss.sum() / len(train_loader), train_bpd.sum() / len(train_loader)))

    return train_loss, train_bpd


def evaluate(train_loader, val_loader, model, model_sample, args, testing=False, file=None, epoch=0):
    model.eval()

    loss_type = 'bpd'

    def analyse(data_loader, plot=False):
        bpds = []
        batch_idx = 0
        with torch.no_grad():
            for data, _ in data_loader:
                batch_idx += 1

                if args.cuda:
                    data = data.cuda()

                data = data.view(-1, *args.input_size)

                loss, batch_bpd, bpd_per_prior, pz, z, pys, ys, ldj = \
                    model(data)
                loss = torch.mean(loss).item()
                batch_bpd = torch.mean(batch_bpd).item()

                bpds.append(batch_bpd)

        bpd = np.mean(bpds)

        with torch.no_grad():
            if not testing and plot:
                x_sample = model_sample.sample(n=100)

                try:
                    plot_reconstructions(
                        x_sample, bpd, loss_type, epoch, args)
                except:
                    print('Not plotting')

        return bpd

    bpd_train = analyse(train_loader)
    bpd_val = analyse(val_loader, plot=True)

    with open(file, 'a') as ff:
        msg = 'epoch {}\ttrain bpd {:.3f}\tval bpd {:.3f}\t'.format(
                epoch,
                bpd_train,
                bpd_val)
        print(msg, file=ff)

    loss = bpd_val * np.prod(args.input_size) * np.log(2.)
    bpd = bpd_val

    file = None

    # Compute log-likelihood
    with torch.no_grad():
        if testing:
            test_data = val_loader.dataset.data_tensor

            if args.cuda:
                test_data = test_data.cuda()

            print('Computing log-likelihood on test set')

            model.eval()

            log_likelihood = analyse(test_data)

        else:
            log_likelihood = None
            nll_bpd = None

        if file is None:
            if testing:
                print('====> Test set loss: {:.4f}'.format(loss))
                print('====> Test set log-likelihood: {:.4f}'.format(log_likelihood))

                print('====> Test set bpd (elbo): {:.4f}'.format(bpd))
                print('====> Test set bpd (log-likelihood): {:.4f}'.format(log_likelihood/
                                                                           (np.prod(args.input_size) * np.log(2.))))

            else:
                print('====> Validation set loss: {:.4f}'.format(loss))
                print('====> Validation set bpd: {:.4f}'.format(bpd))
        else:
            with open(file, 'a') as ff:
                if testing:
                    print('====> Test set loss: {:.4f}'.format(loss), file=ff)
                    print('====> Test set log-likelihood: {:.4f}'.format(log_likelihood), file=ff)

                    print('====> Test set bpd: {:.4f}'.format(bpd), file=ff)
                    print('====> Test set bpd (log-likelihood): {:.4f}'.format(log_likelihood /
                                                                               (np.prod(args.input_size) * np.log(2.))),
                          file=ff)

                else:
                    print('====> Validation set loss: {:.4f}'.format(loss), file=ff)
                    print('====> Validation set bpd: {:.4f}'.format(loss / (np.prod(args.input_size) * np.log(2.))),
                          file=ff)

    if not testing:
        return loss, bpd
    else:
        return log_likelihood, nll_bpd
