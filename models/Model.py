import torch
import models.generative_flows as generative_flows
import numpy as np
from models.utils import Base
from .priors import Prior
from optimization.loss import compute_loss_array
from coding.coder import encode_sample, decode_sample


class Normalize(Base):
    def __init__(self, args):
        super().__init__()
        self.n_bits = args.n_bits
        self.variable_type = args.variable_type
        self.input_size = args.input_size

    def forward(self, x, ldj, reverse=False):
        domain = 2.**self.n_bits

        if self.variable_type == 'discrete':
            # Discrete variables will be measured on intervals sized 1/domain.
            # Hence, there is no need to change the log Jacobian determinant.
            dldj = 0
        elif self.variable_type == 'continuous':
            dldj = -np.log(domain) * np.prod(self.input_size)
        else:
            raise ValueError

        if not reverse:
            x = (x - domain / 2) / domain
            ldj += dldj
        else:
            x = x * domain + domain / 2
            ldj -= dldj

        return x, ldj


class Model(Base):
    """
    The base VAE class containing gated convolutional encoder and decoder
    architecture. Can be used as a base class for VAE's with normalizing flows.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.variable_type = args.variable_type
        self.distribution_type = args.distribution_type

        n_channels, height, width = args.input_size

        self.normalize = Normalize(args)

        self.flow = generative_flows.GenerativeFlow(
            n_channels, height, width, args)

        self.n_bits = args.n_bits

        self.z_size = self.flow.z_size

        self.prior = Prior(self.z_size, args)

    def dequantize(self, x):
        if self.training:
            x = x + torch.rand_like(x)
        else:
            # Required for stability.
            alpha = 1e-3
            x = x + alpha + torch.rand_like(x) * (1 - 2 * alpha)

        return x

    def loss(self, pz, z, pys, ys, ldj):
        batchsize = z.size(0)
        loss, bpd, bpd_per_prior = \
            compute_loss_array(pz, z, pys, ys, ldj, self.args)

        for module in self.modules():
            if hasattr(module, 'auxillary_loss'):
                loss += module.auxillary_loss() / batchsize

        return loss, bpd, bpd_per_prior

    def forward(self, x):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log
         det jacobian is zero for a plain VAE (without flows), and z_0 = z_k.
        """
        # Decode z to x.

        assert x.dtype == torch.uint8

        x = x.float()

        ldj = torch.zeros_like(x[:, 0, 0, 0])
        if self.variable_type == 'continuous':
            x = self.dequantize(x)
        elif self.variable_type == 'discrete':
            pass
        else:
            raise ValueError

        x, ldj = self.normalize(x, ldj)

        z, ldj, pys, ys = self.flow(x, ldj, pys=(), ys=())

        pz, z, ldj = self.prior(z, ldj)

        loss, bpd, bpd_per_prior = self.loss(pz, z, pys, ys, ldj)

        return loss, bpd, bpd_per_prior, pz, z, pys, ys, ldj

    def inverse(self, z, ys):
        ldj = torch.zeros_like(z[:, 0, 0, 0])
        x, ldj, pys, py = \
            self.flow(z, ldj, pys=[], ys=ys, reverse=True)

        x, ldj = self.normalize(x, ldj, reverse=True)

        x_uint8 = torch.clamp(x, min=0, max=255).to(
                torch.uint8)

        return x_uint8

    def sample(self, n):
        z_sample = self.prior.sample(n)

        ldj = torch.zeros_like(z_sample[:, 0, 0, 0])
        x_sample, ldj, pys, py = \
            self.flow(z_sample, ldj, pys=[], ys=[], reverse=True)

        x_sample, ldj = self.normalize(x_sample, ldj, reverse=True)

        x_sample_uint8 = torch.clamp(x_sample, min=0, max=255).to(
                torch.uint8)

        return x_sample_uint8

    def encode(self, x):
        batchsize = x.size(0)
        _, _, _, pz, z, pys, ys, _ = self.forward(x)

        pjs = list(pys) + [pz]
        js = list(ys) + [z]

        states = []

        for b in range(batchsize):
            state = None
            for pj, j in zip(pjs, js):
                pj_b = [param[b:b+1] for param in pj]
                j_b = j[b:b+1]

                state = encode_sample(
                    j_b, pj_b, self.variable_type,
                    self.distribution_type, state=state)
                if state is None:
                    break

            states.append(state)

        return states

    def decode(self, states):
        def decode_fn(states, pj):
            states = list(states)
            j = []

            for b in range(len(states)):
                pj_b = [param[b:b+1] for param in pj]

                states[b], j_b = decode_sample(
                    states[b], pj_b, self.variable_type,
                    self.distribution_type)

                j.append(j_b)

            j = torch.cat(j, dim=0)
            return states, j

        states, z = self.prior.decode(states, decode_fn=decode_fn)

        ldj = torch.zeros_like(z[:, 0, 0, 0])

        x, ldj = self.flow.decode(z, ldj, states, decode_fn=decode_fn)

        x, ldj = self.normalize(x, ldj, reverse=True)
        x = x.to(dtype=torch.uint8)

        return x
