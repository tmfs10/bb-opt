
import numpy as np
from scipy.stats import kendalltau

import hsic
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.distributions as tdist

import reparam_trainer as reparam
from tqdm import tnrange, trange

class Qz(nn.Module):
    def __init__(self, num_latent, prior_std):
        super(Qz, self).__init__()
        self.mu_z = Parameter(torch.zeros(num_latent))
        self.std_z = Parameter(torch.ones(num_latent)*prior_std)
        
    def forward(self, e):
        return self.mu_z.unsqueeze(0) + e*self.std_z.unsqueeze(0)
    
class DnaNN(nn.Module):
    def __init__(self, n_inputs, num_latent, num_hidden, activation):
        super(DnaNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs + num_latent, num_hidden),
            #nn.Linear(n_inputs, num_hidden),
            getattr(nn, activation)(),
            nn.Linear(num_hidden, 1),
        )
        
    def forward(self, x, z, resize_at_end=False, batch_size=0):
        x = torch.cat([x, z], dim=1)
        x = self.net(x)
        return x.view(-1)
        
        
def get_model_nn(
    prior_mean,
    prior_std,
    n_inputs,
    num_latent,
    device='cuda',
    n_hidden=100,
    activation="ReLU",
):
    model = DnaNN(n_inputs, num_latent, n_hidden, activation)
    print(model)
    model = model.to(device)

    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(
                module.weight.data, mode="fan_out", nonlinearity="relu"
            )

    model.apply(init_weights)
    model.train()
    
    qz = Qz(num_latent, prior_std).to('cuda')
    qz.train()
    
    mu_e = torch.zeros(num_latent, requires_grad=False).to(device)
    std_e = torch.ones(num_latent, requires_grad=False).to(device)
    
    e_dist = tdist.Normal(mu_e + prior_mean, std_e*prior_std)
    
    return model, qz, e_dist

def train(
        params,
        batch_size,
        lr,
        num_epochs,
        hsic_lambda,
        num_latent_samples,
        data,
        model,
        qz,
        e_dist,
        jupyter=False,
):
    losses = []
    kl_losses = []
    hsic_losses = []

    corrs = []
    val_corrs = []

    train_X, train_Y, val_X, val_Y = data

    N = train_X.shape[0]
    print("training:")
    print(str(batch_size) + " batch_size")
    print(str(num_epochs) + " num_epochs")

    model_parameters = []
    for m in [model, qz]:
        model_parameters += list(m.parameters())
    batches, optim = reparam.init_train(batch_size, lr, model_parameters, train_X, train_Y)
    num_batches = len(batches)-1
    print(str(num_batches) + " num_batches")

    if jupyter:
        progress = tnrange(num_epochs)
    else:
        progress = trange(num_epochs)

    for epoch_iter in progress:
        for bi in range(num_batches):
            bs = batches[bi]
            be = batches[bi+1]
            bN = be-bs
            if bN <= 0:
                continue

            bX = train_X[bs:be]
            bY = train_Y[bs:be]

            for k in range(1):
                e = reparam.generate_prior_samples(num_latent_samples, e_dist)
                loss, log_prob_loss, kl_loss, hsic_loss, _, _ = reparam.compute_loss(params, batch_size, num_latent_samples, bX, bY, model, qz, e, hsic_lambda=hsic_lambda)
                losses += [log_prob_loss]
                kl_losses += [kl_loss]
                hsic_losses += [hsic_loss]

                optim.zero_grad()
                loss.backward()
                optim.step()
        
        e = reparam.generate_prior_samples(num_latent_samples, e_dist)    
        preds = reparam.predict(train_X, model, qz, e)
        preds = preds.view(-1, num_latent_samples).mean(1)
        assert preds.shape == train_Y.shape, str(preds.shape) + " == " + str(train_Y.shape)
            
        corrs.append(kendalltau(preds, train_Y)[0])

        preds = reparam.predict(val_X, model, qz, e).mean(1).view(-1)
        assert preds.shape == val_Y.shape, str(preds.shape) + " == " + str(val_Y.shape)
        
        val_corr = kendalltau(preds, val_Y)[0]

        val_corrs.append(val_corr)
        progress.set_description(f"Corr: {val_corr:.3f}")
        if jupyter:
            progress.set_postfix({'hsic_loss' : hsic_losses[-1], 'kl_loss' : kl_losses[-1], 'log_prob_loss' : losses[-1]})

    return [losses, kl_losses, hsic_losses, corrs, val_corrs], optim
