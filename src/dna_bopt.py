
import numpy as np
from scipy.stats import kendalltau

import hsic
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.distributions as tdist

import reparam_trainer as reparam
from tqdm import tnrange, trange

from deep_ensemble_sid import (
    NNEnsemble,
    RandomNN,
)

def sample_uniform(out_size):
    z = np.zeros((8*out_size,4))
    z[range(8*out_size),np.random.randint(4,size=8*out_size)] = 1
    out_data = torch.from_numpy(z).view((-1,32)).float().cuda()
    return out_data

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


def get_model_nn_ensemble(
    num_inputs,
    batch_size,
    num_models,
    num_hidden,
    device,
    sigmoid_coeff,
    extra_random: bool = False,
):
    model = NNEnsemble.get_model(num_inputs, batch_size, num_models, num_hidden, device, sigmoid_coeff=sigmoid_coeff, extra_random=extra_random)
    model = model.to(device)
    return model

def train_ensemble(
        params,
        batch_size,
        num_epochs,
        data,
        model_ensemble,
        optim,
        unseen_reg="normal",
        gamma=0.0,
        choose_type="last",
        jupyter=False,
):
    train_X, train_Y, val_X, val_Y = data
    N = train_X.shape[0]
    print("training:")
    print(str(batch_size) + " batch_size")
    print(str(num_epochs) + " num_epochs")
    print(str(gamma) + " gamma")

    num_batches = N//batch_size+1
    batches = [i*batch_size  for i in range(num_batches)] + [N]

    corrs = []
    val_corrs = []
    val_nlls = []
    val_mses = []
    train_nlls = []
    train_mses = []

    if jupyter:
        progress = tnrange(num_epochs)
    else:
        progress = trange(num_epochs)

    best_nll = float('inf')
    best_model = None

    for epoch_iter in progress:
        model_ensemble.train()
        for bi in range(num_batches):
            bs = batches[bi]
            be = batches[bi+1]
            bN = be-bs
            if bN <= 0:
                continue

            bX = train_X[bs:be]
            bY = train_Y[bs:be]

            optim.zero_grad()
            means, variances = model_ensemble(bX)
            assert means.shape[1] == bY.shape[0], "%s[1] == %s[0]" % (str(mean.shape[1]), str(bY.shape[0]))
            nll = NNEnsemble.compute_negative_log_likelihood(
                    bY,
                    means, 
                    variances, 
                    return_mse=False)

            loss = nll
            train_nlls += [nll.item()]
            mse = torch.sqrt(torch.mean((means-bY)**2)).item()
            train_mses += [mse]

            if unseen_reg != "normal":
                assert gamma > 0
                out_data = sample_uniform(bN)
                means_o, variances_o = model_ensemble(out_data)

                if unseen_reg == "maxvar":
                    var = means_o.var(dim=0).mean()
                    loss -= gamma*var
                elif unseen_reg == "defmean":
                    nll = NNEnsemble.compute_negative_log_likelihood(default_mean, means_o, variances_o)
                    loss += gamma*nll

            loss.backward()
            optim.step()

        model_ensemble.eval()
        with torch.no_grad():
            means, variances = model_ensemble(train_X)
            if choose_type == "train":
                nll = NNEnsemble.compute_negative_log_likelihood(
                        train_Y, 
                        means, 
                        variances, 
                        return_mse=False)
                nll = nll.item()

                if nll < best_nll:
                    best_nll = nll
                    best_model = model_ensemble.state_dict()
            means = means.mean(0)
            assert means.shape == train_Y.shape, "%s == %s" % (str(means.shape), str(val_Y.shape))
            corr = kendalltau(means, train_Y)[0]
            corrs += [corr]

            means, variances = model_ensemble(val_X)
            nll = NNEnsemble.compute_negative_log_likelihood(
                    val_Y,
                    means,
                    variances,
                    return_mse=False)
            nll = nll.item()
            mse = torch.sqrt(torch.mean((means-val_Y)**2)).item()
            val_nlls += [nll]
            val_mses += [mse]
            if choose_type == "val" and nll < best_nll:
                best_nll = nll
                best_model = model_ensemble.state_dict()

            means = means.mean(0)
            assert means.shape == val_Y.shape, "%s == %s" % (str(means.shape), str(val_Y.shape))
            val_corr = kendalltau(means, val_Y)[0]
            val_corrs += [val_corr]
            progress.set_description(f"Corr: {val_corr:.3f}")

    if choose_type in ("val", "train"):
        if best_model is not None:
            model_ensemble.load_state_dict(best_model)
        else:
            assert num_epochs == 0
            corrs = [0]
            val_corrs = [0]
            train_nlls = [0]
            train_mses = [0]

    return [corrs, train_nlls, train_mses, val_corrs, val_nlls, val_mses], optim

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
