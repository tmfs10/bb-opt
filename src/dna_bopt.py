
import numpy as np
from scipy.stats import kendalltau

import hsic
import torch
import torch.nn as nn
import sys
import hsic
import copy
from torch.nn.parameter import Parameter
import torch.distributions as tdist
import non_matplotlib_utils as utils
import ops
import bayesian_opt as bopt
from scipy.stats import kendalltau, pearsonr

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


def unseen_data_loss(
    model_ensemble,
    means,
    sample_input,
    bN,
    unseen_reg,
    gamma,
):
    loss = 0
    if unseen_reg != "normal":
        out_data = sample_input(bN)
        means_o, variances_o = model_ensemble(out_data) # (num_samples, num_points)

        if unseen_reg == "maxvar":
            var_mean = means_o.var(dim=0).mean()
            loss -= gamma*var_mean
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
            invar = means.var(dim=0).mean()
            loss -= gamma*invar
        elif unseen_reg == "maxinstd":
            instd = means.std(dim=0).mean()
            loss -= gamma*instd
        elif unseen_reg == "maxinoutstd":
            std_mean = means_o.std(dim=0).mean()
            instd = means.std(dim=0).mean()
            loss -= gamma*0.5*(instd+std_mean)
        elif unseen_reg == "maxout_minin":
            var = means_o.var(dim=0).mean()
            loss -= gamma*var
            invar = means.var(dim=0).mean()
            loss += gamma*invar
        elif unseen_reg == "maxout_minin_std":
            std = means_o.std(dim=0).mean()
            loss -= gamma*std
            instd = means.std(dim=0).mean()
            loss += gamma*instd
        elif unseen_reg == "maxdpp":
            assert False, unseen_reg + " not implemented"
            M = means_o
            mean = M.mean(dim=0, keepdim=True)
            std = M.std(dim=0, keepdim=True)
            M -= mean
            M /= std
            n = means_o.shape[1]
            covar = torch.mm(M.transpose(0, 1), M)/(n-1)
            dpp = torch.logdet(covar)
            print('covar:', covar.mean().item(), dpp.item(), torch.diag(covar).mean().item())
            #loss -= gamma*dpp
        elif unseen_reg == "mincorr":
            corr = ops.corrcoef(means_o.transpose(0, 1))
            corr_mean = torch.tril(corr, diagonal=-1).mean()
            loss += gamma*corr_mean
        elif unseen_reg == "maxhsic":
            M = means_o
            mean = M.mean(dim=0, keepdim=True)
            std = M.std(dim=0, keepdim=True)
            M = M-mean
            M = M/std
            kernel_fn = getattr(hsic, "two_vec_" + params.hsic_kernel_fn)
            kernels = kernel_fn(M, M)  # shape (n=num_samples, n, 1)
            total_hsic = hsic.total_hsic(kernels.repeat([1, 1, 2])).view(-1)
            #print('hsic:', total_hsic.item())
            loss -= gamma*total_hsic[0]
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
            _, nll = NNEnsemble.report_metric(
                    Y,
                    preds.mean(dim=0),
                    preds.var(dim=0),
                    return_mse=False)
            stats['nll'] = nll

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


def train_ensemble(
    params,
    batch_size,
    num_epochs,
    data,
    model_ensemble,
    optim,
    unseen_reg="normal",
    gamma=0.0,
    choose_type="val",
    normalize_fn=None,
    val_frac=0.1,
    early_stopping=10,
    adv_alpha=1.0,
    adv_epsilon=0.0,
    predict_info_models=None,
    hsic_custom_kernel=None,
    reset_rng_state=None,
    jupyter=False,
    ood_val=0.0,
):
    if reset_rng_state is not None:
        cur_rng_state = ops.get_rng_state()
        ops.set_rng_state(reset_rng_state)
        model_ensemble.reset_parameters()
        reset_rng_state = ops.get_rng_state()
        ops.set_rng_state(cur_rng_state)

    adv_train = adv_epsilon > 1e-9
    train_X, train_Y, X, Y = data
    N = train_X.shape[0]
    assert val_frac >= 0.01
    assert val_frac <= 0.9

    if ood_val > 1e-9:
        assert ood_val <= 0.5, "val_frac is %0.3f. validation set cannot be larger than train set" % (ood_val)
        sorted_idx = torch.sort(train_Y, descending=True)[1]
        val_N = N * ood_val
        idx = torch.arange(N)
        train_idx = idx[sorted_idx[val_N:]]
        val_idx = idx[sorted_idx[:val_N]]
    else:
        train_idx, val_idx, _ = utils.train_val_test_split(N, [1-val_frac, val_frac])

    assert val_idx.shape[0] > 0
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

            means, variances = model_ensemble(bX)

            optim.zero_grad()
            assert means.shape[1] == bY.shape[0], "%s[1] == %s[0]" % (str(mean.shape[1]), str(bY.shape[0]))
            nll = NNEnsemble.compute_negative_log_likelihood(
                    bY,
                    means, 
                    variances, 
                    return_mse=False)

            if adv_train:
                assert False, 'Untested'
                negative_log_likelihood = NNEnsemble.compute_negative_log_likelihood(
                    y, 
                    means, 
                    variances, 
                    return_mse=False,
                )

                grad = torch.autograd.grad(
                    negative_log_likelihood, bX, retain_graph=False
                )[0]
                x = bX.detach() + self.adversarial_epsilon * torch.sign(grad)

                loss = adv_alpha*nll + (1-adv_alpha) * negative_log_likelihood
            else:
                loss = nll

            train_nlls += [nll.item()]
            mse = torch.sqrt(torch.mean((means-bY)**2)).item()
            train_mses += [mse]

            loss += unseen_data_loss(
                    model_ensemble,
                    means,
                    sample_uniform,
                    bN,
                    unseen_reg,
                    gamma,
                    )

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
                #sys.exit(1)

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
            means = means.mean(0)
            assert means.shape == val_Y.shape, "%s == %s" % (str(means.shape), str(val_Y.shape))
            val_corr = kendalltau(means, val_Y)[0]
            if choose_type == "val" and nll < best_nll:
                time_since_last_best_epoch = 0
                best_nll = nll
                best_model = copy.deepcopy(model_ensemble.state_dict())
                best_optim = copy.deepcopy(optim.state_dict())

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

    return [corrs, train_nlls, train_mses, train_std, val_corrs, val_nlls, val_mses, val_std, [best_nll]], optim, reset_rng_state


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
