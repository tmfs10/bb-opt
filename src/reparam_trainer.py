
import torch
import sys
import torch.nn as nn
import torch.distributions as tdist

def gaussian_kl(mean, std, prior_std, device):
    var = std**2
    prior_var = torch.tensor(prior_std**2, device=device)
    k = mean.shape[0]
    return 0.5 * torch.sum( var * 1./prior_var + 1./prior_var*(mean**2) - k - torch.log(var) + k*torch.log(prior_var) )
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
    z = z.repeat([N, 1])

    X = expand_to_sample(X, num_samples)
    X = torch.cat([X, z], dim=1)
    output = model(X)

    return output

def predict(X, model, qz, e, device='cuda'):
    num_samples = e.shape[0]
    output = predict_no_resize(X, model, qz, e, device)
    return output.detach().view(-1, num_samples, 2)

def compute_loss(params, X, Y, model, qz, e):
    N = X.shape[0]
    num_samples = params.num_samples
    batch_size = params.batch_size

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
    loss = 0
    for bi in range(num_batches):
        bs = batches[bi]
        be = batches[bi+1]
        bN = be-bs

        bX = X[bs:be]
        bY = Y[bs:be]
        bX = bX.to(params.device)
        bY = bY.to(params.device)

        bY = expand_to_sample(bY, num_samples)

        output = predict_no_resize(X, model, qz, e)
        mu = output[:, 0]
        std = output[:, 1]

        assert mu.shape[0] == std.shape[0]

        if params.output_dist_std > 0:
            std[:] = params.output_dist_std
        output_dist = params.output_dist_fn(mu, std)

        log_prob_loss += -torch.mean(output_dist.log_prob(bY))/num_batches
        #log_prob_loss += torch.mean((bY-mu)**2)/num_batches
        loss += log_prob_loss
        muYhat += [mu]
        stdYhat += [std]
    kl_loss += gaussian_kl(qz.mu_z, qz.std_z, params.prior_std, params.device)
    loss += kl_loss

    muYhat = torch.cat(muYhat, dim=0)
    stdYhat = torch.cat(stdYhat, dim=0)
    return loss, log_prob_loss.item(), kl_loss.item(), muYhat.view(N, -1), stdYhat.view(N, -1)

def generate_prior_samples(num_samples, e_dist, device='cuda'):
    e = []
    for si in range(num_samples):
        e += [e_dist.sample().to(device)]
    e = torch.stack(e, dim=0)
    return e

def train(params, X, Y, model, qz, e_dist):
    assert X.shape[0] == Y.shape[0]

    N = X.shape[0]
    num_batches = N//params.batch_size

    model_parameters = model.parameters() + qz.parameters()
    batches, optim = init_train(params, model_parameters, X, Y)

    for epoch_iter in params.num_epochs:
        for bi in num_batches:
            bs = batches[bi]
            be = batches[bi+1]
            bN = be-bs

            bX = X[bs:be]
            bY = Y[bs:be]

            e = generate_prior_samples(params.num_samples, e_dist)
            loss, log_prob_loss, kl_loss, _, _ = reparam.compute_loss(params, bX, bY, model, qz, e)
            train_losses += [log_prob_loss]
            train_kl_losses += [kl_loss]

            optim.zero_grad()
            loss.backward()
            optim.step()
    return mu_z, std_z


class Qz(nn.Module):
    def __init__(self, num_latent):
        super(Qz, self).__init__()
        self.mu_z = torch.zeros(num_latent)
        self.std_z = torch.ones(num_latent)
        
    def forward(self, e):
        return self.mu_z.unsqueeze(0) + e*self.std_z.unsqueeze(0)
        
def get_model_nn(n_inputs: int = 512, num_latent: int = 20, device='cpu'):
    device = device or "cpu"
    N_HIDDEN = 100
    NON_LINEARITY = "ReLU"

    model = nn.Sequential(
        nn.Linear(n_inputs + num_latent, N_HIDDEN),
        getattr(nn, NON_LINEARITY)(),
        nn.Linear(N_HIDDEN, 2),
    ).to(device)

    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(
                module.weight.data, mode="fan_out", nonlinearity="relu"
            )

    model.apply(init_weights)
    model.train()
    
    qz = Qz(num_latent)
    
    mu_e = torch.zeros(num_latent, requires_grad=False)
    std_e = torch.ones(num_latent, requires_grad=False)
    
    e_dist = tdist.Normal(mu_e + params.prior_mean, std_e*params.prior_std)
    
    return model, qz, e_dist
