
import torch
import torch.nn as nn
import torch.distributions as tdist

def gaussian_kl(mean, std):
    var = std**2
    return 0.5 * ((1+torch.log(var)) - mean**2 - var)

def train(params, X, Y, model, zN):
    assert X.shape[0] == Y.shape[0]

    N = X.shape[0]
    num_batches = N//batch_size
    batches = [i*batch_size  for i in range(num_batches)] + [N]

    mu_e = torch.zeros(zN, requires_grad=False)
    std_e = torch.ones(zN, requires_grad=False)

    mu_z = torch.zeros(zN, requires_grad=True)
    std_z = torch.ones(zN, requires_grad=True)

    std_normal = tdist.Normal(mu_e + params.prior_mean, std_e*params.prior_std)

    optim = torch.optim.Adam(model.parameters() + [mu_z, std_z], lr=params.lr)

    for epoch_iter in params.num_epochs:
        for bi in num_batches:
            bs = batches[bi]
            be = batches[bi+1]
            bN = be-bs+1

            bX = X[bs:be]
            bY = Y[bs:be]

            e = []
            for si in range(params.num_samples):
                e += [std_normal.sample()]
            e = torch.cat(dim=0)

            z = mu_z + e * std_z

            z = z.repeat([1, bN]).view(-1, e.shape[1])
            bX = bX.repeat([params.num_samples] + [1]*(len(bX.shape)-1))
            bY = bY.repeat([params.num_samples] + [1]*(len(bY.shape)-1))

            output = model(bX)
            mu = output[:bN]
            std = output[bN:]

            assert mu.shape[0] == std.shape[0]

            output_dist = tdist.Normal(mu, std)

            loss = output_dist.log_prob(bY)/num_samples + gaussian_kl(mu_z, std_z)

            optim.zero_grad()
            loss.backward()
            optim.step()

