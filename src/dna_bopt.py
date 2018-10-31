
import numpy as np
from scipy.stats import kendalltau

import hsic
import torch
import torch.nn as nn
import copy
from torch.nn.parameter import Parameter
import torch.distributions as tdist
import non_matplotlib_utils as utils

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
        normalize_fn=None,
        val_frac=0.1,
        early_stopping=10,
        jupyter=False,
):
    train_X, train_Y = data
    N = train_X.shape[0]
    assert val_frac >= 0.01
    assert val_frac <= 0.9

    train_idx, val_idx, _ = utils.train_val_test_split(N, [1-val_frac, val_frac])

    val_X = train_X[val_idx]
    train_X = train_X[train_idx]
    val_Y = train_Y[val_idx]
    train_Y = train_Y[train_idx]

    if normalize_fn is not None:
        mean = train_Y.mean()
        std = train_Y.std()
        train_Y = normalize_fn(train_Y, mean, std, exp=torch.exp)
        val_Y = normalize_fn(val_Y, mean, std, exp=torch.exp)

    N = train_X.shape[0]
    print("training:")
    print("%d num_train" % (N))
    print("%d num_val" % (val_X.shape[0]))
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
    train_std = []
    val_std = []

    if jupyter:
        progress = tnrange(num_epochs)
    else:
        progress = trange(num_epochs)

    best_nll = float('inf')
    best_model = None
    best_optim = None

    time_since_last_best_epoch = 0
    for epoch_iter in progress:
        time_since_last_best_epoch += 1
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
            train_std += [means.std(0).mean().item()]
            if choose_type == "train":
                _, nll = NNEnsemble.report_metric(
                        train_Y, 
                        means, 
                        variances, 
                        return_mse=False)
                nll = nll.item()

                if nll < best_nll:
                    time_since_last_best_epoch = 0
                    best_nll = nll
                    best_model = copy.deepcopy(model_ensemble.state_dict())
                    best_optim = copy.deepcopy(optim.state_dict())
            means = means.mean(0)
            assert means.shape == train_Y.shape, "%s == %s" % (str(means.shape), str(val_Y.shape))
            corr = kendalltau(means, train_Y)[0]
            corrs += [corr]

            means, variances = model_ensemble(val_X)
            val_std += [means.std(0).mean().item()]
            _, nll = NNEnsemble.report_metric(
                    val_Y,
                    means,
                    variances,
                    return_mse=False)
            nll = nll.item()
            mse = torch.sqrt(torch.mean((means-val_Y)**2)).item()
            val_nlls += [nll]
            val_mses += [mse]
            if choose_type == "val" and nll < best_nll:
                time_since_last_best_epoch = 0
                best_nll = nll
                best_model = copy.deepcopy(model_ensemble.state_dict())
                best_optim = copy.deepcopy(optim.state_dict())

            means = means.mean(0)
            assert means.shape == val_Y.shape, "%s == %s" % (str(means.shape), str(val_Y.shape))
            val_corr = kendalltau(means, val_Y)[0]
            val_corrs += [val_corr]
            progress.set_description(f"Corr: {val_corr:.3f}")

        if early_stopping > 0 and choose_type in ("val", "train") and time_since_last_best_epoch > early_stopping:
            assert epoch_iter >= early_stopping
            break

    if choose_type in ("val", "train"):
        if best_model is not None:
            model_ensemble.load_state_dict(best_model)
            optim.load_state_dict(best_optim)
        else:
            assert num_epochs == 0
    if num_epochs == 0:
        corrs = [-1]
        train_nlls = [-1]
        train_mses = [-1]
        train_std = [-1]
        val_corrs = [-1]
        val_nlls = [-1]
        val_mses = [-1]
        val_std = [-1]

    return [corrs, train_nlls, train_mses, train_std, val_corrs, val_nlls, val_mses, val_std], optim


def one_hot_list_to_number(inputs, data=None):
    assert len(inputs.shape) >= 3
    if data is None:
        data = []
        data_dict = {}
    arr = np.arange(inputs.shape[-1], dtype=np.int32)+1
    for i in range(inputs.shape[0]):
        num = 0
        rev = 0
        for j in range(inputs.shape[-2]):
            digit = int(np.dot(arr, inputs[i, j]))
            num *= (j+1)
            num += digit
            rev *= (j+1)
            rev += arr[4-digit]

        data += [num]
        data_dict[num] = i
        data_dict[rev] = i
    return data, data_dict


def prob_to_number(inputs, ack_inputs):
    assert len(inputs.shape) == 3
    data = set()

    arr = np.arange(inputs.shape[-1], dtype=np.int32)+1
    rev_compl_mapping = [4, 3, 2, 1]
    for i in range(inputs.shape[0]):
        while True:
            num = 0
            rev = 0
            for j in range(inputs.shape[-2]):
                digit = np.random.choice(arr, p=inputs[i, j])
                num *= (j+1)
                num += digit
                rev *= (j+1)
                rev += arr[4-digit]
            if num not in ack_inputs and rev not in ack_inputs and num not in data and rev not in data:
                data.add(num)
                break
    return data


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
