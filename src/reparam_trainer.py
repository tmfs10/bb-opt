
import torch
import sys
import torch.nn as nn
import torch.distributions as tdist
import random
import numpy as np
from bb_opt.src import utils
from bb_opt.src import hsic


def gaussian_kl(mean, std, prior_std, device):
    assert std.ndimension() == 1
    var = std**2
    prior_var = torch.tensor(prior_std**2, device=device)
    k = mean.shape[0]
    return 0.5 * (torch.sum(var * 1./prior_var) + 1./prior_var*torch.sum(mean**2) - k + torch.sum(-torch.log(var) + torch.log(prior_var)))
    #return 0.5 * torch.sum((1+torch.log(var)) - mean**2 - var)


def init_train(params, model_parameters, X, Y):
    assert X.shape[0] == Y.shape[0]

    N = X.shape[0]
    num_batches = N//params.batch_size
    batches = [i*params.batch_size  for i in range(num_batches)] + [N]

    optim = torch.optim.Adam(model_parameters, lr=params.lr)

    return batches, optim


def expand_to_sample(X, num_samples):
    X = X.unsqueeze(1)
    X = X.repeat([1] + [num_samples] + [1]*(len(X.shape)-2)).view([-1]+list(X.shape[2:]))
    return X


def predict_no_resize(X, model, qz, e, device='cuda'):
    model = model.to(device)
    X = X.to(device)
    qz = qz.to(device)
    e = e.to(device)

    num_samples = e.shape[0]
    assert num_samples > 0
    N = X.shape[0]
    z = qz(e)
    output = model(X, z)

    return output, z


def predict(X, model, qz, e, device='cuda'):
    num_samples = e.shape[0]
    output, z = predict_no_resize(X, model, qz, e, device)
    return output.detach().view(-1, num_samples, 2)


def compute_loss(params, X, Y, model, qz, e, hsic_lambda=0):
    do_hsic = hsic_lambda > 1e-9
    N = X.shape[0]
    num_samples = params.num_samples
    batch_size = params.batch_size
    assert num_samples == e.shape[0]

    if N > batch_size:
        N = X.shape[0]
        num_batches = N//batch_size
        batches = [i*batch_size  for i in range(num_batches)] + [N]
    else:
        num_batches = 1
        batches = [0, N]

    muYhat = []
    stdYhat = []
    log_prob_loss = 0
    kl_loss = 0
    hsic_loss = torch.tensor(0., device=params.device)
    loss = 0
    for bi in range(num_batches):
        bs = batches[bi]
        be = batches[bi+1]
        bN = be-bs

        bX = X[bs:be]
        bY = Y[bs:be]
        bX = bX.to(params.device)
        bY = bY.to(params.device)

        bY = utils.collated_expand(bY, num_samples)

        output, z = predict_no_resize(X, model, qz, e)
        assert z.shape[0] == num_samples
        mu = output[:, 0]
        std = output[:, 1]

        assert mu.shape[0] == std.shape[0]

        if params.output_dist_std > 0:
            std[:] = params.output_dist_std
        output_dist = params.output_dist_fn(mu, std)

        log_prob_loss += -torch.mean(output_dist.log_prob(bY))/num_batches
        #log_prob_loss += torch.mean((bY-mu)**2)/num_batches

        if do_hsic:
            assert z is not None
            z_kernels = hsic.two_vec_mixrq_kernels(z, z) # (n, n, 1)

            m = N
            random_d = torch.tensor(np.random.choice(N, size=m), device=params.device)

            if m < N:
                mu2 = mu.view(N, num_samples)[random_d, :].transpose(0, 1)
            else:
                mu2 = mu.view(N, num_samples).transpose(0, 1)
            mu_kernels = hsic.dimwise_mixrq_kernels(mu2).permute([2, 0, 1]).unsqueeze(-1) # (m, n, n, 1)
            z_kernels = z_kernels.unsqueeze(0).repeat([m, 1, 1, 1]) # (m, n, n, 1)
            kernels = torch.cat([mu_kernels, z_kernels], dim=-1)
            total_hsic = torch.mean(hsic.total_hsic_parallel(kernels))
            hsic_loss += total_hsic/num_batches

            """
            for di in random_d:
                s = di*num_samples
                e = (di+1)*num_samples
                mu2 = mu[s:e]
                mu_kernels = hsic.two_vec_mixrq_kernels(mu2, mu2)
                kernels = torch.cat([mu_kernels, z_kernels], dim=-1)
                total_hsic = hsic.total_hsic(kernels)
                #hsic_loss_vec[di] = total_hsic
                hsic_loss += total_hsic/(num_batches*len(random_d))
            """

        loss += -hsic_lambda*hsic_loss
        loss += log_prob_loss
        muYhat += [mu]
        stdYhat += [std]

    kl_loss += gaussian_kl(qz.mu_z, qz.std_z, params.prior_std, params.device)
    loss += kl_loss

    muYhat = torch.cat(muYhat, dim=0)
    stdYhat = torch.cat(stdYhat, dim=0)
    return loss, log_prob_loss.item(), kl_loss.item(), hsic_loss.item(), muYhat.view(N, -1), stdYhat.view(N, -1)

def generate_prior_samples(num_samples, e_dist, device='cuda'):
    e = []
    for si in range(num_samples):
        e += [e_dist.sample().to(device)]
    e = torch.stack(e, dim=0)
    return e

def train_model_reparam(model, X, Y, n_epochs, batch_size):
    assert X.shape[0] == Y.shape[0]

    model, qz, e_dist, params = model

    N = X.shape[0]
    num_batches = N // batch_size

    model_parameters = model.parameters() + qz.parameters()
    batches, optim = init_train(params, model_parameters, X, Y)

    for epoch_iter in n_epochs:
        for bi in num_batches:
            bs = batches[bi]
            be = batches[bi+1]
            bN = be-bs

            bX = X[bs:be]
            bY = Y[bs:be]

            e = generate_prior_samples(params.num_samples, e_dist)
            loss, log_prob_loss, kl_loss, hsic_loss, _, _ = compute_loss(params, bX, bY, model, qz, e, hsic_lambda=20.0)

            optim.zero_grad()
            loss.backward()
            optim.step()
    return mu_z, std_z


class Qz(nn.Module):
    def __init__(self, num_latent, prior_std=1):
        super(Qz, self).__init__()
        self.mu_z = torch.zeros(num_latent)
        self.std_z = torch.ones(num_latent) * prior_std

    def forward(self, e):
        return self.mu_z.unsqueeze(0) + e * self.std_z.unsqueeze(0)


class DnaNN(nn.Module):
    def __init__(self, n_inputs, num_latent, num_hidden, activation):
        super(DnaNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs + num_latent, num_hidden),
            #nn.Linear(n_inputs, num_hidden),
            getattr(nn, activation)(),
            nn.Linear(num_hidden, 2),
        )

    def forward(self, x, z, resize_at_end=False):
        assert x.ndimension() == 2
        num_samples = z.shape[0]
        N = x.shape[0]

        x = utils.collated_expand(x, num_samples)
        z = z.repeat([N, 1])
        x = torch.cat([x, z], dim=1)

        x = self.net(x)

        if resize_at_end:
            x = x.view([N, num_samples]).transpose()
        return x


def get_model_reparam(
    n_inputs: int = 512,
    num_latent: int = 20,
    prior_mean: float = 0.0,
    prior_std: float = 1.0,
    device='cpu',
    batch_size = None,
    ):
    N_HIDDEN = 100
    NON_LINEARITY = "ReLU"

    model = DnaNN(n_inputs, num_latent, N_HIDDEN, NON_LINEARITY).to(device)

    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(
                module.weight.data, mode="fan_out", nonlinearity="relu"
            )

    model.apply(init_weights)
    model.train()

    qz = Qz(num_latent, prior_std).to(device)
    qz.train()

    mu_e = torch.zeros(num_latent, requires_grad=False).to(device)
    std_e = torch.ones(num_latent, requires_grad=False).to(device)

    e_dist = tdist.Normal(mu_e + prior_mean, std_e * prior_std)

    Params = namedtuple('params', [
        'lr',
        'num_latents',
        'output_dist_std',
        'output_dist_fn',
        'prior_mean',
        'prior_std',
        'num_epochs',
        'num_samples',
        'batch_size',
        'device',
        'exp_noise_samples'
    ])

    params = Params(
        batch_size=100,
        num_latents=20,
        output_dist_std=0.01,
        output_dist_fn=tdist.Normal,
        num_samples=10,
        exp_noise_samples=2,
        lr=1e-3,
        prior_mean=0.,
        prior_std=1.,
        device=device,
        num_epochs=1000
    )

    return model, qz, e_dist, params
