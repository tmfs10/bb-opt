
import torch
import torch.nn as nn
import torch.distributions as tdist

def gaussian_kl(mean, std):
    var = std**2
    return 0.5 * ((1+torch.log(var)) - mean**2 - var)

def init_train(params, X, Y)
    assert X.shape[0] == Y.shape[0]

    N = X.shape[0]
    num_batches = N//params.batch_size
    batches = [i*batch_size  for i in range(num_batches)] + [N]

    optim = torch.optim.Adam(model.parameters() + [mu_z, std_z], lr=params.lr)

    return batches, optim

def compute_loss(params, X, Y, optim, model, qz, e):
    N = X.shape[0]

    if N > params.batch_size:
        N = X.shape[0]
        num_batches = N//params.batch_size
        batches = [i*batch_size  for i in range(num_batches)] + [N]
    else:
        num_batches = 1
        batches = [0, N]

    muYhat = []
    stdYhat = []
    log_prob_loss = 0
    kl_loss = 0
    for bi in num_batches:
        bs = batches[bi]
        be = batches[bi+1]
        bN = be-bs+1

        bX = X[bs:be]
        bY = Y[bs:be]
    
        z = qz(e)
        z = z.repeat([bN, 1])

        bX = bX.to(params.device)
        bY = bX.to(params.device)

        bX = bX.unsqueeze(1)
        bX = bX.repeat([1] + [params.num_samples] + [1]*(len(bX.shape)-1)).view(-1, bX.shape[2:])
        bY = bY.unsqueeze(1)
        bY = bY.repeat([1] + [params.num_samples] + [1]*(len(bY.shape)-1)).view(-1, bY.shape[2:])

        output = model(bX)
        mu = output[:bN]
        std = output[bN:]

        assert mu.shape[0] == std.shape[0]

        output_dist = params.output_dist_fn(mu, std)

        log_prob_loss += output_dist.log_prob(bY)/num_samples
        kl_loss += gaussian_kl(mu_z, std_z)
        loss += log_prob_loss + kl_loss
        muYhat += [mu]
        stdYhat += [std]

    muYhat = torch.cat(muYhat, dim=0)
    stdYhat = torch.cat(stdYhat, dim=0)
    return loss, log_prob_loss.item(), kl_loss.item(), muYhat.view(N, -1), stdYhat.view(N, -1)


def train(params, X, Y, model, qz, e_dist):
    assert X.shape[0] == Y.shape[0]

    N = X.shape[0]
    num_batches = N//params.batch_size

    batches, optim = init_train(params, X, Y)

    for epoch_iter in params.num_epochs:
        for bi in num_batches:
            bs = batches[bi]
            be = batches[bi+1]
            bN = be-bs+1

            bX = X[bs:be]
            bY = Y[bs:be]

            e = []
            for si in range(params.num_samples):
                e += [e_dist.sample()]
            e = torch.cat(dim=0)

            loss, _, _, _, _ = compute_loss(params, bX, bY, optim, model, qz, e)


    return mu_z, std_z
