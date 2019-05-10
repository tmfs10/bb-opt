
import numpy as np
from scipy.stats import kendalltau

import hsic
import random
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import sys
import hsic
import copy
from torch.nn.parameter import Parameter
import torch.distributions as tdist
import non_matplotlib_utils as utils
import ops
import pprint
import bayesian_opt as bopt
from scipy.stats import kendalltau, pearsonr
import gpu_utils

import reparam_trainer as reparam
from tqdm import tnrange, trange

from deep_ensemble_sid import (
    NNEnsemble,
    RandomNN,
    ResnetEnsemble2,
)

def nvidia_smi():
    return gpu_utils.utils.nvidia_smi().split("\n")[4]

def dna_sample_uniform(out_size, sampling_info=None):
    z = np.zeros((8*out_size,4))
    z[range(8*out_size),np.random.randint(4,size=8*out_size)] = 1
    out_data = torch.from_numpy(z).view((-1,32)).float().cuda()
    return out_data

def image_sample_uniform(out_size, rng=None):
    if rng is not None:
        cur_rng = ops.get_rng_state()
        ops.set_rng_state(rng)

    z = np.random.randint(256, size=(out_size,3,32,32))

    if rng is not None:
        rng = ops.get_rng_state()
        ops.set_rng_state(cur_rng)

    out_data = torch.from_numpy(z).float().cuda()
    with torch.no_grad():
        out_data /= 255.
    return out_data, rng

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


class AckEmbeddingKernel(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.Tanh):
        super(AckEmbeddingKernel, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                activation(),
                nn.Linear(out_dim, out_dim),
                activation(),
                )

    def forward(self, x):
        ret = self.net(x) # (num_points, out_dim)
        ret = ret.mean(dim=0)
        #print('Ack embedding:', ret)
        return ret
        

class PredictInfoKernel(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.Tanh):
        super(PredictInfoKernel, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim), 
                activation(),
                nn.Linear(out_dim, out_dim), 
                activation(),
                )

    def forward(self, x):
        return self.net(x)



class StddevNetwork(nn.Module):
    def __init__(self, in_dim):
        super(StddevNetwork, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(in_dim, 100),
                nn.Tanh(),
                nn.Linear(100, 1),
                nn.Tanh(),
                )

    def forward(self, x):
        return (self.net(x))



class NllNetwork(nn.Module):
    def __init__(self, in_dim):
        super(NllNetwork, self).__init__()
        self.net = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.net(x)


class PredictInfoModels(nn.Module):
    def __init__(
        self, 
        in_dim,
        ack_emb_kernel_dim,
        num_models,
        predict_kernel_dim,
        predict_stddev,
        predict_mmd,
        predict_nll,
    ):
        super(PredictInfoModels, self).__init__()

        self.ack_emb_kernel = AckEmbeddingKernel(in_dim, ack_emb_kernel_dim)
        #self.predict_info_kernel = PredictInfoKernel(in_dim+num_models+ack_emb_kernel_dim, predict_kernel_dim)
        self.predict_info_kernel = PredictInfoKernel(in_dim+ack_emb_kernel_dim, predict_kernel_dim)
        #self.predict_info_kernel = PredictInfoKernel(in_dim, predict_kernel_dim)
        self.predict_stddev = predict_stddev
        self.predict_mmd = predict_mmd
        self.predict_nll = predict_nll

        self.module_list = [self.ack_emb_kernel, self.predict_info_kernel]

        self.predict_models = {
                'std' : None,
                'mmd' : None,
                'nll' : None,
                }
        if predict_stddev:
            self.predict_models['std'] = StddevNetwork(predict_kernel_dim)
        if predict_mmd:
            assert False, "Not implemented"
        if predict_nll:
            self.predict_models['nll'] = NllNetwork(predict_kernel_dim)

        self.module_list += [p for _, p in self.predict_models.iteritems() if p is not None]
        self.module_list = nn.ModuleList(self.module_list)

        self.optim = None
        self.idx = None

    def forward(self, x, y, ack_x):
        ack_emb = self.ack_emb_kernel(ack_x) # (out_dim)
        ack_emb = ack_emb.unsqueeze(0).repeat([x.shape[0], 1])
        predictor_x = torch.cat(
                #[x, y.transpose(0, 1), ack_emb], # (num_points, num_samples)
                [x, ack_emb], # (num_points, num_samples)
                #[x], # (num_points, num_samples)
                dim=1)
        Xemb = self.predict_info_kernel(predictor_x) # (num_points, out_dim)
        stats_pred = {}
        for stat_name in self.predict_models:
            if self.predict_models[stat_name] is not None:
                stats_pred[stat_name] = self.predict_models[i](Xemb).view(-1)
        return stats_pred

    def init_opt(
        self,
        params,
        train_idx,
        num_total_points,
    ):
        self.optim = torch.optim.Adam(
                list(self.parameters()),
                lr=3e-3)
        self.idx = ops.range_complement(num_total_points, train_idx)

    def sample_points(self, n):
        idx = torch.randint(self.idx.shape[0], (n,)).long()
        idx = self.idx[idx]
        #idx = self.idx[:n]
        return idx


class OodPredModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(OodPredModel, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(input_size, 100),
                nn.ReLU(),
                nn.Linear(100, output_size),
                nn.ReLU(),
                )

    def forward(self, x):
        return self.net(x)


        
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


def ensemble_forward(model, X, batch_size, jupyter=False, progress_bar=True):
    N = X.shape[0]
    num_batches = N//batch_size+1
    batches = [i*batch_size  for i in range(num_batches)] + [N]

    if progress_bar:
        if jupyter:
            progress = tnrange(num_batches)
        else:
            progress = trange(num_batches)
        progress.set_description(f"ens_for")
    else:
        progress = range(num_batches)

    out_means = []
    out_vars = []
    for bi in progress:
        bs = batches[bi]
        be = batches[bi+1]
        bN = be-bs
        if bN <= 0:
            continue

        out = model(X[bs:be])
        out_means += [out[0].transpose(0, 1)]
        out_vars += [out[1].transpose(0, 1)]

    out_means = torch.cat(out_means, dim=0).transpose(0, 1)
    out_vars = torch.cat(out_vars, dim=0).transpose(0, 1)

    return out_means.contiguous(), out_vars.contiguous()


def get_model_nn_ensemble(
    num_inputs,
    num_models,
    num_hidden,
    device,
    sigmoid_coeff,
    separate_mean_var=False,
    extra_random=False,
    mu_prior=None,
    std_prior=None,
):
    model = NNEnsemble.get_model(
            num_inputs, 
            num_models, 
            num_hidden, 
            device=device, 
            sigmoid_coeff=sigmoid_coeff, 
            extra_random=extra_random,
            separate_mean_var=separate_mean_var,
            mu_prior=mu_prior,
            std_prior=std_prior,
            )
    model = model.to(device)
    return model


def langevin_sampling(
    params,
    x,
    loss_fn, # this shld be the maxvar loss
    xi_dist=None,
):
    for i in range(params.langevin_num_iter):
        x.requires_grad = True
        loss = loss_fn(x)
        loss = loss.sum()
        loss.backward()
        #print(loss.item())

        x.requires_grad = False
        if xi_dist is not None:
            xi = xi_dist.sample(sample_shape=torch.Size([x.shape[0]])).to(params.device).detach()
            x += -params.langevin_lr*x.grad.detach() + math.sqrt(2*params.langevin_lr*1./params.langevin_beta)*xi
        else:
            x += -params.langevin_lr*x.grad.detach()
        x.grad.zero_()

    return x


def unseen_data_loss(
    means_o,
    variances_o,
    unseen_reg,
    gamma,
    means=None,
    o_weighting=None,
):
    loss = 0
    assert unseen_reg != "normal"
    if unseen_reg == "maxvar":
        if o_weighting is not None:
            var_mean = (means_o.var(dim=0)*o_weighting).mean()
        else:
            var_mean = means_o.var(dim=0).mean()
        loss -= gamma*var_mean
    elif unseen_reg == "maxvargeometric":
        var_mean = torch.max(means_o.var(dim=0).mean(), means_o.new_tensor(0.99))
        loss -= gamma/(1-var_mean)
    elif unseen_reg == "maxstd":
        std_mean = means_o.std(dim=0).mean()
        loss -= gamma*std_mean
    elif unseen_reg == "maxstd_std":
        std_std = means_o.std(dim=0).std()
        loss -= gamma*(std_std)
    elif unseen_reg == "maxstd_mean_std":
        std_mean = means_o.std(dim=0).mean()
        std_std = means_o.std(dim=0).std()
        loss -= gamma*0.5*(std_mean + std_std)
    elif unseen_reg == "maxinvar":
        assert means is not None
        invar = means.var(dim=0).mean()
        loss -= gamma*invar
    elif unseen_reg == "maxinoutvar":
        assert means is not None
        var_mean = means_o.var(dim=0).mean()
        loss -= gamma/2*var_mean
        invar = means.var(dim=0).mean()
        loss -= gamma/2*invar
    elif unseen_reg == "maxinstd":
        assert means is not None
        instd = means.std(dim=0).mean()
        loss -= gamma*instd
    elif unseen_reg == "maxinoutstd":
        assert means is not None
        std_mean = means_o.std(dim=0).mean()
        instd = means.std(dim=0).mean()
        loss -= gamma*0.5*(instd+std_mean)
    elif unseen_reg == "defmean":
        nll = NNEnsemble.compute_negative_log_likelihood(default_mean, means_o, variances_o)
        loss += gamma*nll
    else:
        assert False, unseen_reg + " not implemented"

    return loss


def collect_stats_to_predict(
    params,
    Y,
    preds, # (num_samples, num_points)
    std=True,
    mmd=False,
    nll=True,
    hsic=False,
    hsic_custom_kernel=None,
    ack_pred=None, # (num_samples, num_points)
):
    with torch.no_grad():
        stats = {
                'std' : None,
                'mmd' : None,
                'nll' : None,
                'hsic' : None
                }
        if std:
            stats['std'] = preds.std(dim=0)

        if mmd:
            assert False, "Not implemented"

        if nll:
            nll_mixture, nll_single_gaussian = NNEnsemble.report_metric(
                    Y,
                    preds.mean(dim=0),
                    preds.var(dim=0),
                    return_mse=False)
            stats['nll'] = nll_single_gaussian.item()

    if hsic:
        assert hsic_custom_kernel is not None
        assert ack_pred is not None
        ack_emb = hsic_custom_kernel(ack_pred.transpose(0, 1)) # (num_points, emb_dem)
        rand_emb = hsic_custom_kernel(preds.transpose(0, 1)) # (num_points, emb_dem)
        ack_emb = ack_emb.transpose(0, 1)
        rand_emb = rand_emb.transpose(0, 1)

        kernel_fn = getattr(hsic, "dimwise_" + params.hsic_kernel_fn)

        ack_kernels = kernel_fn(ack_emb) # (emb_dem, emb_dem, num_points)
        rand_kernels = kernel_fn(rand_emb) # (emb_dem, emb_dem, num_points)

        num_points = preds.shape[1]
        hsic_loss = 0.
        for point_idx in range(num_points):
            hsic_loss += hsic.hsic_xy(ack_kernels[:, :, point_idx], rand_kernels[:, :, point_idx], normalized=True)
        hsic_loss /= num_points

        stats['hsic'] = hsic_loss

    return stats


def predict_info_loss(
    params,
    model_ensemble,
    predict_info_models,
    bX,
    sX, # sample X
    sY, # sample Y
    old_stats, # stddev, mmd, nll
    hsic=False,
    hsic_custom_kernel=None,
):
    # learn kernel to maximize predictiveness of MMD change/HSIC but 
    # both of those depend on the kernel so the kernel would just 
    # output sth fixed
    #
    # Maybe train the vanilla kernel to be as discriminative as possible
    # in ordering the examples (so maximize the entropy of prediction?) while
    # training the predictor kernel to maximize the predictive capacity
    # of the vanilla kernel

    assert bX.shape[1] == sX.shape[1]
    assert sX.shape[0] == sY.shape[0], "%s[0] == %s[0]" % (sX.shape, sY.shape)
    assert old_stats['std'].shape[0] == sX.shape[0], "%s[0] == %s[0]" % (old_stats['std'].shape, sX.shape)
    assert len(old_stats) == 4, len(old_stats)

    with torch.no_grad():
        X_dist = hsic.sqdist(bX.unsqueeze(1), sX.unsqueeze(1))
        assert X_dist.shape[0] == bX.shape[0], "%s[0] == %s[0]" % (X_dist.shape, sX.shape)
        assert X_dist.shape[1] == sX.shape[0], "%s[1] == %s[1]" % (X_dist.shape, sX.shape)
        assert X_dist.shape[2] == 1, X_dist.shape
        X_dist = X_dist[:, :, 0].mean(dim=0).cpu().numpy()

    with torch.no_grad():
        sY2, _ = model_ensemble(sX) # (num_samples, num_points)
        if hsic:
            ack_pred, _ = model_ensemble(bX) # (num_samples, num_points)
        else:
            ack_pred = None
        new_stats = collect_stats_to_predict(params, sY, sY2, ack_pred=ack_pred)

    stats_pred = predict_info_models(sX, sY2, bX)
    stats_loss = 0.
    for stat_name in old_stats:
        if stat_name in stats_pred and stats_pred[stat_name] is not None:
            diff = (old_stats[stat_name]-new_stats[stat_name])
            assert X_dist.shape[0] == diff.shape[0]

            #p = pearsonr(X_dist, (-diff/old_stats[stat_name]).detach().cpu().numpy())[0]
            #print("dist_p", p)

            vx = diff-torch.mean(diff)
            vy = stats_pred[stat_name]-torch.mean(stats_pred[stat_name])

            vp = torch.sum(vx*vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

            stats_loss += torch.mean((stats_pred[stat_name]-diff)**2)
            #stats_loss -= vp
            if stat_name == 'std':
                #print("diff:", diff)
                #print("stats_pred[stat_name]:", stats_pred[stat_name])
                #print("stats_loss:", stats_loss.item())
                #p = pearsonr(stats_pred[stat_name].detach().cpu().numpy(), diff.detach().cpu().numpy())[0]
                #print("stats_pearson:", p)
                pass
    return stats_loss


def reinit_model(
    params,
    init_model,
    cur_model,
    reset_rng_state=None,
):
    with torch.no_grad():
        if params.ack_model_init_mode == "new_init":
            assert reset_rng_state is not None
            cur_rng_state = ops.get_rng_state()
            ops.set_rng_state(reset_rng_state)

            cur_model.reset_parameters()

            reset_rng_state = ops.get_rng_state()
            ops.set_rng_state(cur_rng_state)
        elif params.ack_model_init_mode == "init_init":
            cur_model = copy.deepcopy(init_model)
        elif params.ack_model_init_mode == "finetune":
            pass
        else:
            assert False, params.ack_model_init_mode + " not implemented"

    return reset_rng_state, cur_model


def langevin_mod_loss(model_ensemble, unseen_reg, density_x=None):
    def loss_fn(X):
        unseen_reg_langevin_mapping = {
                "maxvar" : "maxvar",
                "maxinoutvar" : "maxvar",
                "maxvargeometric" : "maxvargeometric",
                "maxstd" : "maxstd",
                "maxinoutstd" : "maxstd",
                "maxstd_std" : "maxstd_std",
                "maxstd_mean_std" : "maxstd_mean_std",
                }

        means_o, variances_o = model_ensemble(X)

        assert unseen_reg in unseen_reg_langevin_mapping
        var_sum = means_o.var(dim=0).sum()
        loss = var_sum
        #loss = -unseen_data_loss(
        #        means_o,
        #        variances_o,
        #        unseen_reg_langevin_mapping[unseen_reg],
        #        gamma,
        #        means=None,
        #        )

        """
        if density_x is not None:
            assert density_x.shape[1:] == X.shape[1:], "%s[1:] == %s[1:]" % (density_x.shape, X.shape)
            assert len(density_x.shape) >= 2
            if len(density_x.shape) > 2:
                density_x = density_x.view(density_x.shape[0], -1)
                X = X.view(X.shape[0], -1)
            similarity = torch.exp(-ops.sqdist(density_x.unsqueeze(1), X.unsqueeze(1)).mean())
            loss += dist
        """
        return loss
    return loss_fn


def make_validation_set(
    data,
    choose_type,
    num_epoch_iters,
    val_frac,
    ood_val_frac,
    data_split_rng,
):
    train_X, train_Y, val_X, val_Y, X, Y = data
    if "ood" in choose_type:
        assert ood_val_frac > 1e-3
        assert ood_val_frac <= 0.5, "val_frac is %0.3f. validation set cannot be larger than train set" % (ood_val_frac)
        sorted_idx = torch.sort(train_Y, descending=True)[1]
        val_N = int(N * ood_val_frac)
        idx = torch.arange(N)
        train_idx = idx[sorted_idx[val_N:]]
        val_idx = idx[sorted_idx[:val_N]]
    elif "ind" in choose_type:
        train_idx, val_idx, _, data_split_rng = utils.train_val_test_split(
                N, 
                [1-val_frac, val_frac],
                rng=data_split_rng,
                )
    else:
        assert "validation set distribution not found in choose_type=%s" % (choose_type,)

    if num_epoch_iters is None:
        assert val_idx.shape[0] > 0
        val_X = train_X[val_idx]
        train_X = train_X[train_idx]
        val_Y = train_Y[val_idx]
        train_Y = train_Y[train_idx]
        print("%d num_val" % (val_X.shape[0]))

    if normalize_fn is not None:
        mean = train_Y.mean()
        std = train_Y.std()
        train_Y = normalize_fn(train_Y, mean, std, exp=torch.exp)
        if num_epoch_iters is None:
            val_Y = normalize_fn(val_Y, mean, std, exp=torch.exp)


    return train_X, train_Y, val_X, val_Y


def normalize_for_training(
    normalize_fn,
    train_Y,
    val_Y,
    num_epoch_iters,
):
    if normalize_fn is not None:
        mean = train_Y.mean()
        std = train_Y.std()
        train_Y = normalize_fn(train_Y, mean, std, exp=torch.exp)
        if num_epoch_iters is None:
            val_Y = normalize_fn(val_Y, mean, std, exp=torch.exp)
    return train_Y, val_Y


def compute_single_gaussian_baseline(
    train_Y,
):
    train_mean = train_Y.mean()
    train_std = train_Y.std()
    train_normal = tdist.normal.Normal(train_mean, train_std)
    train_baseline_rmse = torch.sqrt(((train_mean-train_Y)**2).mean()).detach().item()

    val_baseline_rmse = val_baseline_nll = None
    if num_epoch_iters is None:
        val_baseline_rmse = torch.sqrt(((train_mean-val_Y)**2).mean()).detach().item()
        val_baseline_nll = -train_normal.log_prob(val_Y).mean().detach().item()

    return train_baseline_rmse, val_baseline_rmse, val_baseline_nll


def make_batches(
    N,
    batch_size,
):
    num_batches = N//batch_size+1
    batches = [i*batch_size  for i in range(num_batches)] + [N]
    return num_batches, batches


def record_stats_ensemble(
    params,
    model,
    X,
    Y,
):
    with torch.no_grad():
        means, variances = model(X)
        means = means.detach()
        variances = variances.detach()

        nll1, nll2 = NNEnsemble.report_metric(
                Y,
                means,
                variances,
                return_mse=False)
        se = (means.mean(dim=0)-Y)**2
        rmse = torch.sqrt(torch.mean(se)).detach().item()
        std = means.std(0).mean().detach().item()
        mean_of_means = means.mean(dim=0).detach()
        assert mean_of_means.shape == Y.shape, "%s == %s" % (mean_of_means.shape, Y.shape)
        kt_corr = kendalltau(mean_of_means.cpu(), Y.cpu())[0]

        return rmse, std, kt_corr, nll1.item(), nll2.item(), mean_of_means


def choose_best_model(
        choose_type,
        nll_criterion,
        kt_corr,
        rmse,
        best_measure,
        best_nll,
        best_kt_corr,
        best_rmse,
        train_mean_of_means,
        val_mean_of_means,
        val_Y,
        ):
    best_model = None
    best_found = False
    if "val" in choose_type:
        if nll_criterion < best_nll:
            best_nll = nll_criterion
            if "nll" in choose_type:
                best_found = True
                best_measure = best_nll
        if kt_corr > best_kt_corr:
            best_kt_corr = kt_corr
            if "kt_corr" in choose_type:
                best_found = True
                best_measure = best_kt_corr
        if rmse < best_rmse:
            best_rmse = rmse
            if "rmse" in choose_type:
                best_found = True
                best_measure = best_rmse
        if "classify" in choose_type:
            kt_labels = [0]*train_mean_of_means.shape[0] + [1]*val_mean_of_means.shape[0]
            mean_preds = torch.cat([train_mean_of_means, val_mean_of_means], dim=0)
            classify_kt_corr = kendalltau(mean_preds, kt_labels)[0]
            if best_measure is None or best_measure < classify_kt_corr:
                best_measure = classify_kt_corr
                best_found = True
        if "bopt" in choose_type:
            max_idx = torch.argmax(val_mean_of_means)
            measure = val_Y[max_idx]
            if best_measure is None or best_measure < measure:
                best_measure = measure
                best_found = True
    else:
        assert False, choose_type + " not implemented"

    return best_found, best_nll, best_kt_corr, best_rmse, best_measure


def knn_density(train_x, x, k=5, true_max=False):
    with torch.no_grad():
        assert train_x.shape[1:] == x.shape[1:], "%s[1:] == %s[1:]" % (train_x.shape, x.shape)
        assert len(x.shape) >= 2
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
            train_x = train_x.view(train_x.shape[0], -1)
        dist = ops.sqdist(x.unsqueeze(1), train_x.unsqueeze(1))
        dist_sort, _ = torch.sort(dist, dim=1)
        if true_max:
            dist_sort = dist_sort[:, k]
        else:
            dist_sort = dist_sort[:, :k].mean(dim=1)
        #dist_sort = (dist_sort-dist_sort.min())/dist_sort.std()
        dist_sort /= dist_sort.max()

        return dist_sort

def knn_density_max(train_x, x, k=5):
    with torch.no_grad():
        assert train_x.shape[1:] == x.shape[1:], "%s[1:] == %s[1:]" % (train_x.shape, x.shape)
        assert len(x.shape) >= 2
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
            train_x = train_x.view(train_x.shape[0], -1)
        dist = ops.sqdist(x.unsqueeze(1), train_x.unsqueeze(1))
        dist_sort, _ = torch.sort(dist, dim=1)
        dist_sort = dist_sort[:, :k].max(dim=1)
        #dist_sort = (dist_sort-dist_sort.min())/dist_sort.std()
        dist_sort /= dist_sort.max()

        return dist_sort

def hsic_max_loss(
    params,
    X,
    unseen_idx,
    old_model,
    cur_model,
):
    n_points = params.infomax_npoints
    assert n_points > 0
    unseen_idx = random.sample(unseen_idx, n_points)
    X = X[unseen_idx]
    old_preds, _ = old_model(X) # (num ensemble members, num points)
    cur_preds, _ = cur_model(X)

    old_dist_matrix = hsic.sqdist(old_preds)
    cur_dist_matrix = hsic.sqdist(cur_preds)

    kernel_fn = getattr(hsic, params.hsic_kernel_fn)

    old_kernels = kernel_fn(old_dist_matrix) # (num ensemble members, num ensemble members, num points)
    cur_kernels = kernel_fn(cur_dist_matrix)

    old_hsic_matrix = hsic.pairwise_hsic(old_kernels, include_self=False, normalized=True)
    cur_hsic_matrix = hsic.pairwise_hsic(cur_kernels, include_self=False, normalized=True)

    diff_hsic_matrix = cur_hsic_matrix-old_hsic_matrix
    penalty_matrix = -torch.tril(F.logsigmoid(diff_hsic_matrix*10), diagonal=-1)
    loss = penalty_matrix.mean()
    #print('hsic_max_loss', loss.item(), diff_hsic_matrix.mean().item())

    return loss


def corr_max_loss(
    params,
    X,
    unseen_idx,
    old_model,
    cur_model,
):
    n_points = params.infomax_npoints
    assert n_points > 0
    unseen_idx = random.sample(unseen_idx, n_points)
    X = X[unseen_idx]
    old_preds, _ = old_model(X) # (num ensemble members, num points)
    cur_preds, _ = cur_model(X)

    old_corr_matrix = ops.corrcoef(old_preds.transpose(0, 1))
    cur_corr_matrix = ops.corrcoef(cur_preds.transpose(0, 1))

    diff_corr_matrix = cur_corr_matrix-old_corr_matrix
    penalty_matrix = -torch.tril(F.logsigmoid(diff_corr_matrix*100), diagonal=-1)
    loss = penalty_matrix.mean()

    return loss


def pairwise_corr_diversity(
    params,
    means_o,
):
    means_o = means_o.transpose(0, 1)
    corr = ops.corrcoef(means_o)
    if params.pairwise_corr_diversity_mean_weighted:
        means_en = means_o.mean(1)
        means_en_product = means_en.unsqueeze(0) * means_en.unsqueeze(1)
        corr = means_en_product * corr
    return corr.mean()


def pairwise_hsic_diversity(
    params,
    means_o,
):
    dist_matrix = hsic.sqdist(means_o)
    kernel_fn = getattr(hsic, params.hsic_kernel_fn)

    kernels = kernel_fn(dist_matrix) # (num ensemble members, num ensemble members, num points)
    hsic_matrix = hsic.pairwise_hsic(kernels, include_self=False, normalized=True)
    return hsic_matrix.mean()


def train_ensemble(
    params,
    batch_size,
    num_epochs,
    data,
    model_ensemble,
    optim,
    choose_type,
    unseen_reg="normal",
    gamma=0.0,
    normalize_fn=None,
    num_epoch_iters=None,
    val_frac=0.1,
    early_stopping=10,
    adv_alpha=1.0,
    adv_epsilon=0.0,
    predict_info_models=None,
    hsic_custom_kernel=None,
    data_split_rng=None,
    jupyter=False,
    ood_val_frac=0.0,
    sample_uniform_fn=None,
    ood_sampling_rng=None,
    unseen_idx=None,
):
    with torch.no_grad():
        old_model_ensemble = None
        if params.infomax_weight > ops._eps:
            old_model_ensemble = copy.deepcopy(model_ensemble)
        do_early_stopping = ("val" in choose_type or "train" in choose_type) and (num_epoch_iters is None)

        adv_train = adv_epsilon > 1e-9
        train_X, train_Y, val_X, val_Y, X, Y = data
        N = train_X.shape[0]
        assert val_frac >= 0.01
        assert val_frac <= 0.9

        if "ood" in choose_type:
            assert ood_val_frac > 1e-3
            assert ood_val_frac <= 0.5, "val_frac is %0.3f. validation set cannot be larger than train set" % (ood_val_frac)
            sorted_idx = torch.sort(train_Y, descending=True)[1]
            val_N = int(N * ood_val_frac)
            idx = torch.arange(N)
            train_idx = idx[sorted_idx[val_N:]]
            val_idx = idx[sorted_idx[:val_N]]
        elif "ind" in choose_type:
            if val_X is None:
                train_idx, val_idx, _, data_split_rng = utils.train_val_test_split(
                        N, 
                        [1-val_frac, val_frac],
                        rng=data_split_rng,
                        )
            else:
                train_idx, _, _, data_split_rng = utils.train_val_test_split(
                        N, 
                        [1., 0.],
                        rng=data_split_rng,
                        )
        else:
            assert "validation set distribution not found in choose_type=%s" % (choose_type,)

        if val_X is None and val_idx.shape[0] > 0:
            val_X = train_X[val_idx]
            val_Y = train_Y[val_idx]
        train_X = train_X[train_idx]
        train_Y = train_Y[train_idx]
        print("%d num_val" % (val_X.shape[0]))

        if normalize_fn is not None:
            mean = train_Y.mean()
            std = train_Y.std()
            train_Y = normalize_fn(train_Y, mean, std, exp=torch.exp)
            if val_X is not None:
                val_Y = normalize_fn(val_Y, mean, std, exp=torch.exp)

        train_mean = train_Y.mean()
        train_std = train_Y.std()
        train_normal = tdist.normal.Normal(train_mean, train_std)
        train_baseline_rmse = torch.sqrt(((train_mean-train_Y)**2).mean()).detach().item()
        if val_X is not None:
            val_baseline_rmse = torch.sqrt(((train_mean-val_Y)**2).mean()).detach().item()
            val_baseline_nll = -train_normal.log_prob(val_Y).mean().detach().item()

        N = train_X.shape[0]
        print("training:")
        print("%d num_train" % (N))
        print(str(batch_size) + " batch_size")
        print(str(num_epochs) + " num_epochs")
        print(str(gamma) + " gamma")

        num_batches = N//batch_size+1
        batches = [i*batch_size  for i in range(num_batches)] + [N]

        train_kt_corrs = []
        val_kt_corrs = []
        train_nlls = [[], []]
        val_nlls = [[], []]
        val_rmses = []
        train_rmses = []
        train_stds = []
        val_stds = []
        best_val_indv_rmse = None

        if params.progress_bar:
            if jupyter:
                progress = tnrange(num_epochs)
            else:
                progress = trange(num_epochs)
        else:
            progress = range(num_epochs)

        best_nll = float('inf')
        best_epoch_iter = -1
        best_kt_corr = -2.
        best_rmse = float('inf')
        best_measure = None
        best_model = None
        best_optim = None

        time_since_last_best_epoch = 0
        logging = None

    for epoch_iter in progress:
        if num_epoch_iters is not None and epoch_iter >= num_epoch_iters:
            break
        time_since_last_best_epoch += 1
        model_ensemble.train()

        for bi in range(num_batches):
            bs = batches[bi]
            be = batches[bi+1]
            bN = be-bs
            ood_data_batch_size = int(math.ceil(bN*params.ood_data_batch_factor))
            if bN <= 0:
                continue

            if isinstance(model_ensemble, NNEnsemble):
                sampling_info = {}
            else:
                sampling_info = {'ood_sampling_rng': ood_sampling_rng, 'train_x': train_X, 'model': model_ensemble}
            with torch.no_grad():
                bX = train_X[bs:be].detach()
                bY = train_Y[bs:be].detach()

                if params.unseen_reg != "normal" and params.sampling_dist == "uniform_bb" and isinstance(model_ensemble, NNEnsemble):
                    temp = bX.view(bX.shape[0], -1)
                    min_px = temp.min(dim=0)[0]
                    max_px = temp.max(dim=0)[0]
                    sampling_info['min_px'] = min_px
                    sampling_info['max_px'] = max_px

            if params.inverse_density_emb_space and isinstance(model_ensemble, ResnetEnsemble2):
                means, variances, conv_emb = model_ensemble(bX, return_fc_input=True)
            else:
                means, variances = model_ensemble(bX)

            if params.sampling_dist == "pom_fc_input":
                assert isinstance(model_ensemble, ResnetEnsemble2)
                hist = [[np.histogram(fc_input[:, i], bins=10, density=False) for i in range(fc_input.shape[1])] for fc_input in conv_emb]
                hist = [[(k[i][0]/float(sum(k[i][0])), k[i][1]) for i in range(len(k))] for k in hist]
                sampling_info['hist'] = hist
            elif params.sampling_dist == "bb_fc_input":
                assert isinstance(model_ensemble, ResnetEnsemble2)
                min_val = []
                max_val = []
                for i in range(len(bb_fc_input)):
                    temp = bb_fc_input[i]
                    min_val += [temp.min(dim=0)[0]]
                    max_val += [temp.max(dim=0)[0]]
                sampling_info['min_val'] = min_val
                sampling_info['max_val'] = max_val

            optim.zero_grad()
            assert means.shape[1] == bY.shape[0], "%s[1] == %s[0]" % (str(mean.shape[1]), str(bY.shape[0]))
            if params.loss_fn == "nll":
                nll = model_ensemble.compute_negative_log_likelihood(
                        bY,
                        means, 
                        variances,
                        return_mse=False)
                loss = nll
            elif params.loss_fn == "mse":
                mse = (bY - means) ** 2
                loss = mse
            else:
                assert False, params.loss_fn + " not implemented"
            loss += model_ensemble.bayesian_ensemble_loss(params.fixed_noise_std)/bN

            if old_model_ensemble is not None:
                info_fn = globals()[params.infotype + '_max_loss']
                loss += params.infomax_weight * info_fn(
                        params,
                        X,
                        unseen_idx,
                        old_model_ensemble,
                        model_ensemble,
                        )

            optim.zero_grad()
            if (unseen_reg != "normal" or params.pairwise_corr_diversity) and gamma > 0.0:
                if isinstance(model_ensemble, ResnetEnsemble2):
                    model_ensemble.freeze_conv()

                out_data = sample_uniform_fn(ood_data_batch_size, sampling_info=sampling_info)

                if params.sampling_space != "fc" and params.inverse_density_emb_space:
                    assert isinstance(model_ensemble, ResnetEnsemble2)
                    out_data_conv_emb = model_ensemble.conv_forward(out_data).detach()

                if params.langevin_sampling and gamma > 0.0:
                    model_ensemble.eval()
                    #num_features = out_data.shape[1]
                    #xi_dist = tdist.Normal(torch.zeros(num_features), torch.ones(num_features))
                    #means_o, variances_o = model_ensemble(out_data) # (num_samples, num_points)
                    out_data = langevin_sampling(
                            params,
                            out_data,
                            langevin_mod_loss(model_ensemble, unseen_reg),
                            #xi_dist,
                            )
                    #means_o, variances_o = model_ensemble(out_data) # (num_samples, num_points)
                    model_ensemble.train()

                    if isinstance(model_ensemble, ResnetEnsemble2):
                        model_ensemble.freeze_conv()

                if params.sampling_space == "fc":
                    assert isinstance(model_ensemble, ResnetEnsemble2)
                    weighting = torch.ones(out_data.shape[1], device=params.device)
                else:
                    weighting = torch.ones(out_data.shape[0], device=params.device)

                if params.inverse_density:
                    with torch.no_grad():
                        assert params.sampling_space != "fc" or params.inverse_density_emb_space
                        if params.inverse_density_emb_space:
                            assert isinstance(model_ensemble, ResnetEnsemble2)
                            assert conv_emb.shape[0] == out_data_conv_emb.shape[0], "%s[0] == %s[0]" % (conv_emb.shape, out_data_conv_emb.shape)
                            temp = []
                            for i_ensemble in range(conv_emb.shape[0]):
                                temp += [knn_density(conv_emb[i_ensemble], out_data_conv_emb[i_ensemble], true_max=params.true_max).view(-1).detach()]
                            temp = torch.stack(temp).mean(0)
                        else:
                            temp = knn_density(bX, out_data, true_max=params.true_max).view(-1).detach()

                        assert temp.shape == weighting.shape, "%s == %s" % (temp.shape, weighting.shape)
                        weighting = temp

                if params.inverse_density_emb_space:
                    assert isinstance(model_ensemble, ResnetEnsemble2)
                    means_o, variances_o = model_ensemble.fc_forward(out_data_conv_emb) # (num_samples, num_points)
                else:
                    means_o, variances_o = model_ensemble(out_data) # (num_samples, num_points)

                if params.pairwise_corr_diversity:
                    ood_loss = gamma * pairwise_corr_diversity(
                            params,
                            means_o,
                            )
                elif params.unseen_reg != "normal":
                    ood_loss = unseen_data_loss(
                            means_o,
                            variances_o,
                            unseen_reg,
                            gamma,
                            means=means,
                            o_weighting=weighting
                            )

                else:
                    assert False, "Not implemented"

                if isinstance(model_ensemble, ResnetEnsemble2):
                    ood_loss.backward(retain_graph=True)
                    model_ensemble.unfreeze_conv()
                else:
                    loss += ood_loss

            if predict_info_models is not None:
                s_idx = predict_info_models.sample_points(params.num_predict_sample_points)
                with torch.no_grad():
                    predictor_preds, _ = model_ensemble(X[s_idx])
                old_stats = collect_stats_to_predict(
                        params,
                        Y[s_idx],
                        predictor_preds,
                        )

                loss += old_stats['hsic']

            loss.backward()

            optim.step()
            torch.cuda.empty_cache()

        if predict_info_models is not None:
            for i in range(150):
                predict_info_models.optim.zero_grad()
                predictor_loss = predict_info_loss(
                        params,
                        model_ensemble,
                        predict_info_models,
                        bX,
                        X[s_idx],
                        Y[s_idx],
                        old_stats,
                        )
                predictor_loss.backward()
                predict_info_models.optim.step()

        model_ensemble.eval()
        with torch.no_grad():
            train_rmse, train_std, train_kt_corr, train_nll1, train_nll2, train_mean_of_means = record_stats_ensemble(params, model_ensemble, train_X, train_Y)
            train_rmses += [train_rmse]
            train_stds += [train_std]
            train_kt_corrs += [train_kt_corr]
            train_nlls[0] += [train_nll1]
            train_nlls[1] += [train_nll2]

            if val_X is not None:
                val_rmse, val_std, val_kt_corr, val_nll1, val_nll2, val_mean_of_means = record_stats_ensemble(params, model_ensemble, val_X, val_Y)
                val_rmses += [val_rmse]
                val_stds += [val_std]
                val_kt_corrs += [val_kt_corr]
                val_nlls[0] += [val_nll1]
                val_nlls[1] += [val_nll2]

                nll_criterion = val_nll2 if params.single_gaussian_test_nll else val_nll1

            if num_epoch_iters is None:
                assert val_X is not None
                best_found, best_nll, best_kt_corr, best_rmse, best_measure = choose_best_model(
                        choose_type,
                        nll_criterion,
                        val_kt_corr,
                        val_rmse,
                        best_measure,
                        best_nll,
                        best_kt_corr,
                        best_rmse,
                        train_mean_of_means,
                        val_mean_of_means,
                        val_Y,
                        )

                if best_found:
                    time_since_last_best_epoch = 0
                    best_epoch_iter = epoch_iter
                    best_model = copy.deepcopy(model_ensemble.state_dict())

                if params.progress_bar:
                    progress.set_description(f"Corr: {val_kt_corr:.3f}")

                if early_stopping > 0 and do_early_stopping and time_since_last_best_epoch > early_stopping:
                    assert epoch_iter >= early_stopping
                    break

    if do_early_stopping and (num_epoch_iters is None):
        if best_model is not None:
            model_ensemble.load_state_dict(best_model)
        else:
            assert num_epochs == 0

    if num_epochs == 0:
        train_kt_corrs = [-1]
        train_nlls = [-1]
        train_rmses = [-1]
        train_stds = [-1]
        val_kt_corrs = [-1]
        train_nlls = [[-1], [-1]]
        val_nlls = [[-1], [-1]]
        val_rmses = [-1]
        val_stds = [-1]

    if num_epoch_iters is None:
        print ('best_nll:', best_nll)
    else:
        print ('end_nll:', float(val_nlls[1][-1]))


    if val_X is not None:
        logging =  [
                {
                'train' : {
                    #'kt_corr': kt_corrs,
                    #'nll': train_nlls,
                    #'rmse': train_rmses,
                    #'std': train_stds,
                    },
                'val' : {
                    'kt_corr': val_kt_corrs,
                    'nll1': val_nlls[0],
                    'nll2': val_nlls[1],
                    'rmse': val_rmses,
                    'std': val_stds,
                    },
                'baseline': {
                    'nll': val_baseline_nll,
                    'rmse': val_baseline_rmse,
                    'train_rmse': train_baseline_rmse,
                    },
                'best': {
                    'nll': best_nll,
                    'kt_corr': best_kt_corr,
                    'epoch_iter': best_epoch_iter,
                    'indv_rmse': best_val_indv_rmse,
                    'measure': best_measure,
                    },
                },
                {
                'train' : {
                    'kt_corr': float(train_kt_corrs[best_epoch_iter]),
                    'nll1': float(train_nlls[0][best_epoch_iter]),
                    'nll2': float(train_nlls[1][best_epoch_iter]),
                    'rmse': float(train_rmses[best_epoch_iter]),
                    'std': float(train_stds[best_epoch_iter]),
                    },
                'val' : {
                    'kt_corr': float(val_kt_corrs[best_epoch_iter]),
                    'nll1': float(val_nlls[0][best_epoch_iter]),
                    'nll2': float(val_nlls[1][best_epoch_iter]),
                    'rmse': float(val_rmses[best_epoch_iter]),
                    'std': float(val_stds[best_epoch_iter]),
                    },
                'baseline': {
                    'nll': val_baseline_nll,
                    'rmse': val_baseline_rmse,
                    'train_rmse': train_baseline_rmse,
                    },
                'best': {
                    'nll': best_nll,
                    'kt_corr': best_kt_corr,
                    'epoch_iter': best_epoch_iter,
                    'indv_rmse': best_val_indv_rmse,
                    'measure': best_measure,
                    },
                },
        ]

    return logging, data_split_rng


def hyper_param_train(
    params,
    model,
    data,
    stage,
    gammas,
    unseen_reg,
    data_split_rng,
    predict_info_models=None,
    sample_uniform_fn=None,
    normalize_fn=None,
    report_zero_gamma=False,
    num_epoch_iters=None,
    unseen_idx=None,
):
    gamma_added = False
    regression = params.num_acks == 0
    if unseen_reg == "normal" and not params.pairwise_corr_diversity:
        gammas = [0.0]
    if report_zero_gamma and 0.0 not in gammas:
        gammas = [0.0] + gammas
        gamma_added = True

    train_X, train_Y, val_X, val_Y, X, Y = data

    do_hyper_param_search = len(gammas) > 1 or num_epoch_iters is None # search for gamma and/or early stopping point
    do_combine_train_val_training = params.combine_train_val and (val_X is None) and not regression 
    best_gamma = None
    if not do_hyper_param_search:
        best_gamma = gammas[0]
        do_combine_train_val_training = True

    best_nll = float('inf')
    best_logging = None
    zero_logging = None

    zero_gamma_nll = None
    zero_gamma_best_epoch_iter = None

    zero_gamma_model = None
    best_gamma_model = None

    train_batch_size = getattr(params, stage + "_train_batch_size")
    train_epochs = getattr(params, stage + "_train_num_epochs")
    lr = getattr(params, stage + "_train_lr")
    l2 = getattr(params, stage + "_train_l2")

    ood_sampling_rng = ops.get_rng_state()

    best_epoch_iter = None

    logging = None
    if do_hyper_param_search:
        for gamma in gammas:
            model_copy = copy.deepcopy(model)
            optim = torch.optim.Adam(list(model_copy.parameters()), lr=lr, weight_decay=l2)
            data_split_rng2 = copy.deepcopy(data_split_rng)
            best_cur_epoch_iter = None
            logging, data_split_rng2 = train_ensemble(
                    params,
                    train_batch_size,
                    train_epochs,
                    data,
                    model_copy,
                    optim,
                    choose_type=params.hyper_search_choose_type,
                    unseen_reg=unseen_reg,
                    gamma=gamma,
                    normalize_fn=normalize_fn,
                    val_frac=params.val_frac,
                    early_stopping=params.early_stopping,
                    data_split_rng=data_split_rng2,
                    ood_val_frac=params.ood_val_frac,
                    sample_uniform_fn=sample_uniform_fn,
                    ood_sampling_rng=copy.deepcopy(ood_sampling_rng),
                    num_epoch_iters=num_epoch_iters,
                    unseen_idx=unseen_idx,
                    )
            torch.cuda.empty_cache()

            found_best = False
            best_cur_epoch_iter = None
            if num_epoch_iters is None:
                val_nll_cur = logging[1]['best']['nll']
            else:
                val_nll_cur = logging[1]['val']['nll2']
            best_cur_epoch_iter = logging[1]['best']['epoch_iter']

            if gamma == 0.0:
                zero_gamma_best_epoch_iter = best_cur_epoch_iter
                zero_gamma_nll = float(val_nll_cur)
                zero_gamma_model = copy.deepcopy(model_copy)
                zero_logging = logging

            if gamma > 0.0 or not gamma_added:
                if val_nll_cur < best_nll:
                    print('new_best_maxvar:', val_nll_cur)
                    best_nll = float(val_nll_cur)
                    found_best = True

                if found_best:
                    best_logging = logging
                    best_gamma = gamma
                    best_epoch_iter = best_cur_epoch_iter
                    best_gamma_model = copy.deepcopy(model_copy)

                if params.gamma_cutoff:
                    if not found_best:
                        break

            #model_copy = copy.deepcopy(zero_gamma_model)
            #optim = torch.optim.Adam(list(model_copy.parameters()), lr=lr/10., weight_decay=l2)

    torch.cuda.empty_cache()
    #print('point 3:', nvidia_smi())

    if do_hyper_param_search:
        data_split_rng = data_split_rng2
        logging = best_logging

        assert logging[0] is not None
        assert logging[1] is not None
        print('logging:', pprint.pformat(logging[1]))

        assert best_epoch_iter is not None
        assert best_gamma is not None
        print('best gamma:', best_gamma)

    if not do_combine_train_val_training:
        assert best_gamma_model is not None
    elif params.hyper_search_choose_type != params.final_train_choose_type:
        assert False, "for paper we aren't doing this option"
        optim = torch.optim.Adam(list(model.parameters()), lr=lr, weight_decay=l2)
        logging, data_split_rng2 = train_ensemble(
                params,
                train_batch_size,
                train_epochs,
                data,
                model_copy,
                optim,
                choose_type=params.final_train_choose_type,
                unseen_reg=unseen_reg,
                gamma=gamma,
                normalize_fn=normalize_fn,
                val_frac=params.val_frac,
                early_stopping=params.early_stopping,
                data_split_rng=data_split_rng2,
                ood_val_frac=params.ood_val_frac,
                sample_uniform_fn=sample_uniform_fn,
                ood_sampling_rng=copy.deepcopy(ood_sampling_rng),
                num_epoch_iters=num_epoch_iters,
                unseen_idx=unseen_idx,
                )
    else:
        zero_gamma_model = None
        if report_zero_gamma:
            if best_gamma != 0.0:
                zero_gamma_model = copy.deepcopy(model)

        assert num_epoch_iters is not None or best_epoch_iter >= 0
        print('combine_train_val')
        optim = torch.optim.Adam(list(model.parameters()), lr=lr, weight_decay=l2)
        #print('point 4:', nvidia_smi())
        logging2, _ = train_ensemble(
                params,
                train_batch_size,
                train_epochs, 
                data,
                model,
                optim,
                choose_type=params.final_train_choose_type,
                unseen_reg=unseen_reg,
                gamma=best_gamma,
                normalize_fn=normalize_fn,
                num_epoch_iters=best_epoch_iter+1 if num_epoch_iters is None else num_epoch_iters,
                sample_uniform_fn=sample_uniform_fn,
                ood_sampling_rng=copy.deepcopy(ood_sampling_rng),
                unseen_idx=unseen_idx,
                )
        torch.cuda.empty_cache()
        best_gamma_model = model
        if logging is None:
            logging2 = logging

        if zero_gamma_model is not None:
            assert best_gamma != 0.0
            assert zero_gamma_best_epoch_iter >= 0
            optim = torch.optim.Adam(list(zero_gamma_model.parameters()), lr=lr, weight_decay=l2)
            zero_logging, _ = train_ensemble(
                    params, 
                    train_batch_size,
                    train_epochs, 
                    data,
                    zero_gamma_model,
                    optim,
                    choose_type=params.final_train_choose_type,
                    unseen_reg="normal",
                    gamma=0.0,
                    normalize_fn=normalize_fn,
                    num_epoch_iters=zero_gamma_best_epoch_iter+1,
                    sample_uniform_fn=None,
                    ood_sampling_rng=copy.deepcopy(ood_sampling_rng),
                    unseen_idx=unseen_idx,
                    )
            torch.cuda.empty_cache()

        #print('point 5:', nvidia_smi())
        print('combine_train_val done')

    return logging, best_gamma, data_split_rng, zero_logging, zero_gamma_model, best_gamma_model

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
    assert not np.any(np.isnan(inputs)), str(inputs)
    assert np.all(np.isfinite(inputs)), str(inputs)
    assert len(inputs.shape) == 3
    data = []

    arr = np.arange(inputs.shape[-1], dtype=np.int32)+1
    rev_compl_mapping = [4, 3, 2, 1]
    for i in range(inputs.shape[0]):
        while True:
            num = 0
            rev = 0
            for j in range(inputs.shape[-2]):
                try:
                    digit = np.random.choice(arr, p=inputs[i, j])
                except Exception as e:
                    print(inputs[i, j])
                num *= (j+1)
                num += digit
                rev *= (j+1)
                rev += arr[4-digit]
            if num not in ack_inputs and rev not in ack_inputs and num not in data and rev not in data:
                data += [num]
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

    if params.progress_bar:
        if jupyter:
            progress = tnrange(num_epochs)
        else:
            progress = trange(num_epochs)
    else:
        progress = range(num_epochs)

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
        if params.progress_bar:
            progress.set_description(f"Corr: {val_corr:.3f}")
            if jupyter:
                progress.set_postfix({'hsic_loss' : hsic_losses[-1], 'kl_loss' : kl_losses[-1], 'log_prob_loss' : losses[-1]})

    return [losses, kl_losses, hsic_losses, corrs, val_corrs], optim
