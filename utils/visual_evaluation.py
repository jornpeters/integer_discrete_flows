from __future__ import print_function
import os
import numpy as np
import imageio
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec


def plot_reconstructions(recon_mean, loss, loss_type, epoch, args):
    if epoch == 1:
        if not os.path.exists(args.snap_dir + 'reconstruction/'):
            os.makedirs(args.snap_dir + 'reconstruction/')
    if loss_type == 'bpd':
        fname = str(epoch) + '_bpd_%5.3f' % loss
    elif loss_type == 'elbo':
        fname = str(epoch) + '_elbo_%6.4f' % loss
    plot_images(args, recon_mean.data.cpu().numpy()[:100], args.snap_dir + 'reconstruction/', fname)


def plot_images(args, x_sample, dir, file_name, size_x=10, size_y=10):
    batch, channels, height, width = x_sample.shape

    print(x_sample.shape)

    mosaic = np.zeros((height * size_y, width * size_x, channels))

    for j in range(size_y):
        for i in range(size_x):
            idx = j * size_x + i

            image = x_sample[idx]

            mosaic[j*height:(j+1)*height, i*height:(i+1)*height] = \
                image.transpose(1, 2, 0)

    # Remove channel for BW images
    mosaic = mosaic.squeeze()

    imageio.imwrite(dir + file_name + '.png', mosaic)

    # fig = plt.figure(figsize=(size_x, size_y))
    # # fig = plt.figure(1)
    # gs = gridspec.GridSpec(size_x, size_y)
    # # gs.update(wspace=0.05, hspace=0.05)
    # gs.update(wspace=0.0, hspace=0.0)

    # for i, sample in enumerate(x_sample):
    #     ax = plt.subplot(gs[i])
    #     plt.axis('off')
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])
    #     ax.set_aspect('equal')
    #     sample = sample.reshape((args.input_size[0], args.input_size[1], args.input_size[2]))
    #     sample = sample.swapaxes(0, 2)
    #     sample = sample.swapaxes(0, 1)
    #     if args.input_size[0] == 1:
    #         sample = sample[:, :, 0]
    #         plt.imshow(sample, cmap='gray', vmin=0, vmax=1)
    #     else:
    #         plt.imshow(sample)

    # plt.savefig(dir + file_name + '.png', bbox_inches='tight')
    # plt.close(fig)

