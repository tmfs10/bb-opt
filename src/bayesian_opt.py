import os
import random
import copy
import gc
import sys
import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import kendalltau, pearsonr
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as get_fingerprint
import torch
from tqdm import tnrange, trange
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pyro
import pyro.optim
import info_measures as info
from pyro.distributions import (
    Normal,
    TransformedDistribution,
    InverseAutoregressiveFlow,
    TorchDistribution,
    Bernoulli,
    Laplace,
    Gamma,
)
from typing import (
    Union,
    Tuple,
    Optional,
    Callable,
    Dict,
    Set,
    List,
    Any,
    Iterable,
    Sequence,
)
import pickle
from bb_opt.src.hsic import (
    dimwise_mixrq_kernels,
    dimwise_mixrbf_kernels,
    precompute_batch_hsic_stats,
    compute_point_hsics,
    total_hsic,
    total_hsic_with_batch,
)
import bb_opt.src.hsic as hsic
from bb_opt.src.knn_mi import estimate_mi
from bb_opt.src import ops
from bb_opt.src.non_matplotlib_utils import get_path
import bb_opt.src.non_matplotlib_utils as utils
from bb_opt.src.bo_model import BOModel
from bb_opt.src.acquisition_functions import AcquisitionFunction, Uniform

def ExtendedGamma(shape, scale, ndimension=0):
    device = "cuda" if scale.is_cuda else "cpu"
    beta = 1 / scale
    sign = pyro.sample("signX", Bernoulli(torch.tensor(0.5, device=device)))
    sign = sign * 2 - 1
    return sign * pyro.sample("egX", Gamma(shape, beta).independent(ndimension))


def laplace_priors(
    model: torch.nn.Module, mean: float = 0., b: float = 1.
) -> Dict[str, TorchDistribution]:
    priors = {}
    for name, param in model.named_parameters():
        priors[name] = Laplace(
            torch.full_like(param, fill_value=mean),
            torch.full_like(param, fill_value=b),
        ).independent(param.ndimension())

    return priors


def extended_gamma_priors(
    model: torch.nn.Module, mean: float = 0., std: float = 1.
) -> Dict[str, TorchDistribution]:
    priors = {}
    for name, param in model.named_parameters():

        def fn(*args, **kwargs):
            return ExtendedGamma(
                torch.full_like(param, fill_value=mean),
                torch.full_like(param, fill_value=std),
                param.ndimension(),
            )

        priors[name] = fn
    return priors


class SpikeSlabNormal(TorchDistribution):
    def __init__(self, param, std1: float = 0.03, std2: float = 2, alpha: float = 0.7):
        super().__init__(event_shape=param.shape)
        self.normal1 = Normal(
            torch.zeros_like(param), torch.ones_like(param) * std1
        ).independent(param.ndimension())

        self.normal2 = Normal(
            torch.zeros_like(param), torch.ones_like(param) * std2
        ).independent(param.ndimension())

        self.alpha = torch.tensor(alpha).to(param.device)
        self.bernoulli = Bernoulli(self.alpha)

        # self.batch_shape = self.normal1.batch_shape
        # self.event_shape = self.normal1.event_shape

    def sample(self, sample_shape=torch.Size([])):
        bernoullis = self.bernoulli.sample(sample_shape)
        bernoulli_inverse = (~bernoullis.byte()).float()
        sampled = bernoullis * self.normal1.sample(
            sample_shape
        ) + bernoulli_inverse * self.normal2.sample(sample_shape)
        print("SAMPLED")
        print(sampled)
        print()
        return sampled

    def log_prob(self, value):
        log_p1 = self.normal1.log_prob(value)
        log_p2 = self.normal2.log_prob(value)

        c = -torch.max(log_p1, log_p2)  # for numerical stability
        log_p = (
            self.alpha * (c + log_p1).exp() + (1 - self.alpha) * (c + log_p2).exp()
        ).log() - c

        return log_p


def spike_slab_priors(
    model: torch.nn.Module, std1: float = 0.03, std2: float = 2., alpha: float = 0.7
) -> Dict[str, TorchDistribution]:
    priors = {}
    for name, param in model.named_parameters():
        priors[name] = SpikeSlabNormal(param, std1, std2, alpha)

    return priors


def laplace_variationals(
    model: torch.nn.Module, mean: float = 0., b: float = 1.
) -> Dict[str, TorchDistribution]:
    variational_dists = {}
    for name, param in model.named_parameters():
        location = pyro.param(f"g_{name}_location", torch.randn_like(param) + mean)
        log_scale = pyro.param(
            f"g_{name}_log_scale", torch.randn_like(param) + np.log(np.exp(b) - 1)
        )
        variational_dists[name] = Laplace(
            location, torch.nn.Softplus()(log_scale)
        ).independent(param.ndimension())
    return variational_dists


def extended_gamma_variationals(
    model: torch.nn.Module, mean: float = 0., std: float = 1.
) -> Dict[str, TorchDistribution]:
    variational_dists = {}
    for name, param in model.named_parameters():
        location = pyro.param(f"g_{name}_location", torch.randn_like(param) + mean)
        log_scale = pyro.param(
            f"g_{name}_log_scale", torch.randn_like(param) + np.log(np.exp(std) - 1)
        )

        def fn(*args, **kwargs):
            return ExtendedGamma(
                location, torch.nn.Softplus()(log_scale), param.ndimension()
            )

        variational_dists[name] = fn


def iaf_variationals(
    model: torch.nn.Module, n_hidden: int
) -> Dict[str, TorchDistribution]:
    """
    Create IAF variational distributions from a standard normal base.
    """

    variational_dists = {}
    for name, param in model.named_paramters():
        location = torch.zeros_like(param)
        scale = torch.ones_like(param)
        base_dist = Normal(location, scale).independent(param.ndimension())

        iaf = InverseAutoregressiveFlow()

        variational_dists[name] = TransformedDistribution(base_dist, [iaf])


def optimize(
    model: BOModel,
    acquisition_func: AcquisitionFunction,
    inputs: np.ndarray,
    labels: Union[pd.Series, np.ndarray],
    top_k_percent: int = 1,
    batch_size: int = 256,
    n_epochs: int = 2,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    exp: Optional = None,
    n_batches: int = -1,
    retrain_every: int = 1,
    partial_train_epochs: int = 10,
    save_key: str = "",
    acquisition_func_greedy: Optional[AcquisitionFunction] = None,
) -> np.ndarray:

    n_batches = n_batches if n_batches != -1 else int(np.ceil(len(labels) / batch_size))
    if save_key and exp:
        save_key += "_" + exp.get_key()[:5]

    if isinstance(labels, pd.Series):
        labels = labels.values

    n_top_k_percent = int(top_k_percent / 100 * len(labels))
    best_idx = set(labels.argsort()[-n_top_k_percent:])
    sum_best = np.sort(labels)[-n_top_k_percent:].sum()
    best_value = np.sort(labels)[-1]

    try:
        sampled_idx = set()
        fraction_best_sampled = []

        for step in range(n_batches):
            if step == 0:
                acquire_samples = Uniform.acquire(
                    model,
                    inputs,
                    labels[list(sampled_idx)],
                    sampled_idx,
                    batch_size,
                )
            else:
                acquire_samples = acquisition_func(
                    model,
                    inputs,
                    labels[list(sampled_idx)],
                    sampled_idx,
                    batch_size,
                )

            if acquisition_func_greedy:
                acquire_samples_greedy = acquisition_func_greedy(
                    model,
                    inputs,
                    labels[list(sampled_idx)],
                    sampled_idx,
                    batch_size,
                )
            else:
                acquire_samples_greedy = acquire_samples

            greedy_sampled_idx = sampled_idx.copy()
            sampled_idx.update(acquire_samples)
            greedy_sampled_idx.update(acquire_samples_greedy)

            fraction_best = len(best_idx.intersection(greedy_sampled_idx)) / len(
                best_idx
            )
            fraction_best_sampled.append(fraction_best)

            if verbose:
                print(f"{step} {fraction_best:.3f}")

            if exp:
                sampled_labels = labels[list(greedy_sampled_idx)]
                sampled_sum_best = (
                    np.sort(sampled_labels)[-n_top_k_percent:].sum()
                )
                sampled_best_value = sampled_labels.max()

                exp.log_metric("fraction_best", fraction_best, step * batch_size)
                exp.log_metric("best_value", sampled_best_value, step * batch_size)
                exp.log_metric(
                    "best_value_ratio",
                    sampled_best_value / best_value,
                    step * batch_size,
                )
                exp.log_metric(
                    "best_values_ratio", sampled_sum_best / sum_best, step * batch_size
                )

            if fraction_best == 1.0:
                break

            if step % retrain_every == 0:
                model.reset()
                model.train_model(
                    inputs[list(sampled_idx)],
                    labels[list(sampled_idx)],
                    n_epochs,
                )
            else:
                model.train_model(
                    inputs[acquire_samples],
                    labels[acquire_samples],
                    partial_train_epochs,
                )

            if save_key:
                save_fname = get_path(
                    __file__, "..", "..", "models_nathan", save_key, step
                )

                model.save_model(save_fname)
        assert (len(sampled_idx) == min(len(labels), batch_size * n_batches)) or (
            fraction_best == 1.0
        )
    except KeyboardInterrupt:
        pass

    return model


def get_model_uniform(*args, **kwargs):
    pass


def acquire_batch_uniform(
    model,
    inputs,
    sampled_labels,
    sampled_idx: Set[int],
    batch_size: int,
    exp: Optional = None,
) -> List[int]:
    unused_idx = [i for i in range(len(inputs)) if i not in sampled_idx]
    np.random.shuffle(unused_idx)
    return unused_idx[:batch_size]


def train_model_uniform(*args, **kwargs):
    pass


def get_model_nn(n_inputs: int = 512, batch_size: int = 200, device: Optional = None):
    device = device or "cpu"

    model = nn.Sequential(
        nn.Linear(n_inputs, N_HIDDEN),
        getattr(nn, NON_LINEARITY)(),
        nn.Linear(N_HIDDEN, 1),
    ).to(device)

    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(
                module.weight.data, mode="fan_out", nonlinearity="relu"
            )

    model.apply(init_weights)
    model.train()
    return model


def acquire_batch_nn_greedy(
    model,
    inputs,
    sampled_labels,
    sampled_idx: Set[int],
    batch_size: int,
    exp: Optional = None,
) -> List[int]:
    # I'm predicting on the entire data set each time even though we only need preds on the
    # un-acquired data; keeping track of the indices is easier this way, though if it's too
    # slow, I'm sure you could change this
    model.eval()
    preds = model(inputs).squeeze()

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    sorted_idx = np.argsort(preds)
    sorted_idx = [i for i in sorted_idx if i not in sampled_idx]
    acquire_samples = sorted_idx[-batch_size:]  # sorted smallest first
    return acquire_samples


def train_model_nn(model, inputs, labels, n_epochs, batch_size):
    data = TensorDataset(inputs, labels)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        for batch in loader:
            inputs, labels = batch
            optimizer.zero_grad()

            predictions = model(inputs).squeeze()
            loss = loss_func(predictions, labels)
            loss.backward()
            optimizer.step()
    model.eval()


def get_model_bnn_laplace(
    n_inputs: int = 512,
    batch_size: int = 200,
    prior_mean: float = 0,
    prior_std: float = 0.05,
    device=None,
):
    device = device or "cpu"
    model = nn.Sequential(
        nn.Linear(n_inputs, N_HIDDEN),
        getattr(nn, NON_LINEARITY)(),
        nn.Linear(N_HIDDEN, 1),
    ).to(device)

    priors = lambda: laplace_priors(model, prior_mean, prior_std)
    variational_dists = lambda: laplace_variationals(model, prior_mean, prior_std)

    bnn_model = make_bnn_model(model, priors, batch_size=batch_size)
    guide = make_guide(model, variational_dists)
    return bnn_model, guide


def get_model_bnn_extended_gamma(
    n_inputs: int = 512,
    batch_size: int = 200,
    prior_mean: float = 0,
    prior_std: float = 0.05,
    device=None,
):
    device = device or "cpu"
    model = nn.Sequential(
        nn.Linear(n_inputs, N_HIDDEN),
        getattr(nn, NON_LINEARITY)(),
        nn.Linear(N_HIDDEN, 1),
    ).to(device)

    priors = lambda: extended_gamma_priors(model, prior_mean, prior_std)
    variational_dists = lambda: extended_gamma_variationals(
        model, prior_mean, prior_std
    )

    bnn_model = make_bnn_model(model, priors, batch_size=batch_size)
    guide = make_guide(model, variational_dists)
    return bnn_model, guide


def acquire_batch_ei(
    model,
    inputs,
    sampled_labels,
    sampled_idx: Set[int],
    batch_size: int,
    exp: Optional = None,
    n_bnn_samples: int = 50,
    **unused_kwargs,
) -> List[int]:
    if isinstance(model, Sequence):  # bnn
        bnn_model, guide = model
        preds = bnn_predict(guide, inputs, n_samples=n_bnn_samples)
        preds = preds.mean(axis=0)
        sorted_idx = np.argsort(preds)
        sorted_idx = [i for i in sorted_idx if i not in sampled_idx]
        acquire_samples = sorted_idx[-batch_size:]  # sorted smallest first
    else:  # deep ensemble
        with torch.no_grad():
            means, variances = model(inputs, individual_predictions=False)

        acquire_samples = [
            i for i in means.sort(descending=True)[1].tolist() if i not in sampled_idx
        ]
        acquire_samples = acquire_samples[:batch_size]

    return acquire_samples


def acquire_batch_pdts(
    model,
    inputs,
    sampled_labels,
    sampled_idx: Set[int],
    batch_size: int,
    exp: Optional = None,
    kernel: Optional[Callable] = None,
    preds_multiplier: float = 2.0,
) -> Set[int]:
    bnn_model, guide = model
    batch = []
    acquirable_idx = list(set(range(len(inputs))).difference(sampled_idx))

    for _ in range(batch_size):
        with torch.no_grad():
            preds = guide()(inputs[acquirable_idx])

        acquire_idx = preds.argmax().item()
        acquire_idx = acquirable_idx[acquire_idx]

        batch.append(acquire_idx)
        acquirable_idx.remove(acquire_idx)

        if not acquirable_idx:
            break

    assert len(batch) == batch_size or not acquirable_idx
    return batch


def acquire_batch_hsic_mean_std(
    model,
    inputs,
    sampled_labels,
    sampled_idx: Set[int],
    batch_size: int,
    n_points_parallel: int = 100,
    exp: Optional = None,
    kernel: Optional[Callable] = None,
    hsic_coeff: float = 150.0,
    preds_multiplier: float = 2.0,
    metric: str = "mu / sigma - hsic",
) -> List[int]:
    n_preds = int(preds_multiplier * batch_size)
    bnn_model, guide = model

    with torch.no_grad():
        preds = [torch.unsqueeze(guide()(inputs).squeeze(), 0) for _ in range(n_preds)]
    preds = torch.cat(preds)

    mean = preds.mean(dim=0)
    std = preds.std(dim=0)

    acquirable_idx = set(range(len(mean))).difference(sampled_idx)

    best_metric = -float("inf")
    best_idx = None

    # pick the first point to maximize `metric` but ignoring hsic
    hsic_term_idx = max(metric.rfind("-"), metric.rfind("+"))
    assert "hsic" in metric[hsic_term_idx:], metric[hsic_term_idx:]

    for idx in acquirable_idx:
        mu = mean[idx]
        sigma = std[idx]
        point_metric = eval(metric[:hsic_term_idx])
        if point_metric > best_metric:
            best_metric = point_metric
            best_idx = idx

    batch = [best_idx]
    acquirable_idx.remove(best_idx)

    while len(batch) < batch_size:
        best_idx = None
        best_batch_metric = -float("inf")

        batch_stats = precompute_batch_hsic_stats(preds, batch, kernel)

        all_hsics = []

        for next_points in torch.tensor(list(acquirable_idx)).split(n_points_parallel):
            hsics = compute_point_hsics(preds, next_points, *batch_stats, kernel)
            idx = hsics.argmin()
            hsic = hsic_coeff * hsics[idx]
            idx = next_points[idx]

            all_hsics.append(hsics.cpu().numpy())

            mu = mean[idx]
            sigma = std[idx]

            batch_metric = eval(metric)

            if batch_metric > best_batch_metric:
                best_batch_metric = batch_metric
                best_idx = idx.item()

        batch.append(best_idx)
        acquirable_idx.remove(best_idx)

        all_hsics = np.concatenate(all_hsics)

        if exp:
            exp.log_multiple_metrics(
                {"hsic_mean": np.mean(all_hsics), "hsic_std": np.std(all_hsics)},
                step=len(sampled_idx) + len(batch),
            )

        if not acquirable_idx:
            break
    return batch


def acquire_batch_hsic_pdts(
    model,
    inputs,
    sampled_labels,
    sampled_idx: Set[int],
    batch_size: int,
    n_points_parallel: int = 100,
    exp: Optional = None,
    kernel: Optional[Callable] = None,
    preds_multiplier: float = 2.0,
    pdts_multiplier: float = 2.0,
) -> List[int]:

    acquirable_idx = acquire_batch_pdts(
        model,
        inputs,
        sampled_labels,
        sampled_idx,
        int(batch_size * pdts_multiplier),
        exp,
        kernel,
    )

    n_preds = int(preds_multiplier * batch_size)
    bnn_model, guide = model

    with torch.no_grad():
        preds = [torch.unsqueeze(guide()(inputs).squeeze(), 0) for _ in range(n_preds)]
    preds = torch.cat(preds)

    batch = [acquirable_idx.pop()]

    for _ in range(batch_size - 1):
        best_idx = None
        min_hsic = float("inf")

        batch_stats = precompute_batch_hsic_stats(preds, batch, kernel)

        all_hsics = []

        for next_points in torch.tensor(list(acquirable_idx)).split(n_points_parallel):
            hsics = compute_point_hsics(preds, next_points, *batch_stats, kernel)
            idx = hsics.argmin()
            hsic = hsics[idx]
            idx = next_points[idx]

            all_hsics.append(hsics.cpu().numpy())

            if hsic < min_hsic:
                min_hsic = hsic
                best_idx = idx.item()

        batch.append(best_idx)
        acquirable_idx.remove(best_idx)

        all_hsics = np.concatenate(all_hsics)

        if exp:
            exp.log_multiple_metrics(
                {
                    "hsic_mean": np.mean(all_hsics),
                    "hsic_std": np.std(all_hsics),
                    "hsic_min": min_hsic,
                },
                step=len(sampled_idx) + len(batch),
            )

        if not acquirable_idx:
            break

    assert (len(batch) == batch_size) or not acquirable_idx, len(batch)
    return batch


def acquire_batch_mves(
    model,
    inputs,
    sampled_labels,
    sampled_idx: Set[int],
    batch_size: int,
    n_points_parallel: int = 100,
    exp: Optional = None,
    kernel: Optional[Callable] = None,
    mi_estimator: str = "HSIC",
    n_max_dist_points: int = 1,
) -> List[int]:
    return acquire_batch_es(
        model,
        inputs,
        sampled_labels,
        sampled_idx,
        batch_size,
        n_points_parallel,
        exp,
        kernel,
        mi_estimator,
        max_value_es=True,
        n_max_dist_points=n_max_dist_points,
    )


def acquire_batch_es(
    model,
    inputs,
    sampled_labels,
    sampled_idx: Set[int],
    batch_size: int,
    n_points_parallel: int = 100,
    exp: Optional = None,
    kernel: Optional[Callable] = None,
    mi_estimator: str = "HSIC",
    max_value_es: bool = False,
    n_max_dist_points: int = 1,
) -> List[int]:

    acquirable_idx = set(range(len(inputs))).difference(sampled_idx)

    if isinstance(model, Sequence):
        n_preds = 250
        bnn_model, guide = model

        with torch.no_grad():
            preds = torch.stack([guide()(inputs).squeeze() for _ in range(n_preds)])
    else:
        with torch.no_grad():
            means, variances = model(inputs)
        preds = means
        n_preds = len(means)

    if max_value_es:
        max_dist = preds.sort(dim=1, descending=True)[0][:, :n_max_dist_points]
    else:
        max_idx = preds.sort(dim=1, descending=True)[1][:, :n_max_dist_points]
        max_dist = inputs[max_idx]

    acquirable_idx = list(acquirable_idx)
    batch = []

    if mi_estimator == "HSIC":
        max_batch_dist = max_dist

        for _ in range(batch_size):
            all_mi = total_hsic_batched(max_batch_dist, preds, kernel, acquirable_idx)
            best_idx = acquirable_idx[all_mi.argmax().item()]
            acquirable_idx.remove(best_idx)
            batch.append(best_idx)

            if not max_value_es:
                # the input distribution has vector samples instead of scalars,
                # so we can't combine everything into one tensor

                if isinstance(max_batch_dist, list):
                    max_batch_dist[1] = torch.cat(
                        (max_batch_dist[1], preds[:, best_idx : best_idx + 1]), dim=-1
                    )
                else:  # set things up the first time
                    max_batch_dist = [max_batch_dist, preds[:, best_idx : best_idx + 1]]

            if exp:
                exp.log_multiple_metrics(
                    {
                        "mi_mean": all_mi.mean().item(),
                        "mi_std": all_mi.std().item(),
                        "mi_max": all_mi.max().item(),
                    },
                    step=len(sampled_idx) + len(batch),
                )
    elif mi_estimator == "LNC":
        if not max_value_es:
            assert False, "LNC not supported for ES, only MVES."

        max_batch_dist = max_dist.cpu().numpy()
        preds = preds.cpu().numpy()

        for _ in range(batch_size):
            all_mi = []

            for idx in acquirable_idx:
                all_mi.append(
                    estimate_mi(
                        np.concatenate(
                            (max_batch_dist, preds[:, idx : idx + 1]), axis=-1
                        ).T
                    )
                )

            all_mi = np.array(all_mi)

            best_idx = acquirable_idx[all_mi.argmax()]
            acquirable_idx.remove(best_idx)
            batch.append(best_idx)
            max_batch_dist = np.concatenate(
                (max_batch_dist, preds[:, best_idx : best_idx + 1]), axis=-1
            )

            if exp:
                exp.log_multiple_metrics(
                    {
                        "mi_mean": all_mi.mean(),
                        "mi_std": all_mi.std(),
                        "mi_max": all_mi.max(),
                    },
                    step=len(sampled_idx) + len(batch),
                )
    else:
        assert False, f"Unrecognized MI estimation method {mi_estimator}."

    if exp:
        if not max_value_es:
            mode_count = (max_idx[:, 0] == max_idx[:, 0].mode()[0]).sum().item()
            selected_count = (max_idx[:, 0] == batch[0]).sum().item()

            exp.log_multiple_metrics(
                {
                    "mode_fraction": mode_count / n_preds,
                    "best_selected_fraction": selected_count / n_preds,
                },
                step=len(sampled_idx) + len(batch),
            )

    assert len(batch) in [
        batch_size,
        len(acquirable_idx),
    ], f"Bad batch length {len(batch)} for batch size {batch_size}."
    return batch


def er_diversity_selection_hsic(
    params,
    preds, #(num_samples (n), num_candidate_points)
    skip_idx,
    num_er=100,
    device = 'cuda',
    ucb=False,
):
    ack_batch_size = params.ack_batch_size
    kernel_fn = getattr(hsic, params.hsic_kernel_fn)

    er = preds.mean(dim=0).view(-1)
    std = preds.std(dim=0).view(-1)

    er += params.ucb*std
    ucb = er

    # extract top ucb idx not already used and their preds
    ucb_sortidx = torch.sort(ucb, descending=True)[1]
    temp = []
    for idx in ucb_sortidx:
        if idx.item() not in skip_idx:
            temp += [idx]
        if len(temp) == num_er:
            break
    ucb_sortidx = torch.stack(temp) # len == m
    ucb = ucb[ucb_sortidx]
    ucb -= ucb.min()
    preds = preds[:, ucb_sortidx]

    chosen_orig_idx = [ucb_sortidx[0].item()]
    chosen_idx = [0]

    ack_batch_dist_matrix = hsic.sqdist(preds[:, 0].unsqueeze(-1)) # (n, n, len(choose_idx))
    self_kernel_matrix = kernel_fn(ack_batch_dist_matrix).detach().repeat([1, 1, 2]) # (n, n, 2)
    normalizer = torch.log(hsic.total_hsic(self_kernel_matrix))

    # precomputation for normalizer
    m = ucb_sortidx.shape[0]
    all_dist_matrix = hsic.sqdist(preds) # (n, n, m)
    all_kernel_matrix = kernel_fn(all_dist_matrix) # (n, n, m)
    permuted_all_kernel_matrix = all_kernel_matrix.permute([2, 0, 1]).unsqueeze(-1) # (m, n, n, 1)
    self_kernel_matrix = permuted_all_kernel_matrix.repeat([1, 1, 1, 2]) # (m, n, n, 2)
    hsic_var = hsic.total_hsic_parallel(self_kernel_matrix)
    hsic_logvar = torch.log(hsic_var).view(-1)

    while len(chosen_idx) < ack_batch_size:
        kernel_matrix = all_kernel_matrix[:, :, chosen_idx].unsqueeze(-1).repeat([1, 1, 1, m]).permute([3, 0, 1, 2]) # (m, n, n, len(chosen_idx))
        kernel_matrix = torch.cat([kernel_matrix, permuted_all_kernel_matrix], dim=-1)
        hsic_covar = hsic.total_hsic_parallel(kernel_matrix).detach()

        normalizer = torch.exp((hsic_logvar + hsic_logvar[chosen_idx].sum())/(len(chosen_idx)+1)) # doesn't double count as we exclude any idx already in choose_idx

        factor = ucb.std()/hsic_covar.std()
        hsic_corr = hsic_covar * factor
        hsic_corr = ucb - params.diversity_coeff*hsic_corr
        #print('hsic_corr:', hsic_corr)

        _, sort_idx = torch.sort(hsic_corr, descending=True)

        for idx in sort_idx:
            idx = int(idx.item())
            if idx not in chosen_idx:
                break

        choice = ucb_sortidx[idx]
        chosen_idx += [idx]
        chosen_orig_idx += [choice.item()]
        ack_batch_dist_matrix += all_dist_matrix[:, :, idx:idx+1]

    return list(chosen_orig_idx)


def er_diversity_selection_detk(
    params,
    preds, #(num_samples, num_candidate_points)
    skip_idx,
    num_ucb=100,
    device = 'cuda',
    do_correlation=True,
    add_I=True,
    do_kernel=False,
):
    ack_batch_size = params.ack_batch_size

    er = preds.mean(dim=0).view(-1)
    std = preds.std(dim=0).view(-1)

    er += params.ucb*std
    ucb = er

    ucb_sortidx = torch.sort(ucb, descending=True)[1]
    temp = []
    for idx in ucb_sortidx:
        if idx.item() not in skip_idx:
            temp += [idx]
        if len(temp) == num_ucb:
            break
    ucb_sortidx = torch.stack(temp) # len == m
    ucb = ucb[ucb_sortidx]
    ucb -= ucb.min()
    ucb = ucb.cpu().numpy()

    if do_kernel:
        preds = preds[:, ucb_sortidx]
        kernel_fn = getattr(hsic, params.hsic_kernel_fn)
        dist_matrix = hsic.sqdist(preds.transpose(0, 1).unsqueeze(1)) # (m, m, 1)
        covar_matrix = kernel_fn(dist_matrix)[:, :, 0] # (m, m)
    else:
        preds = preds[:, ucb_sortidx]
        covar_matrix = np.cov(preds)
        covar_matrix = torch.FloatTensor(covar_matrix, device=device) # (num_candidate_points, num_candidate_points)

    if add_I:
        covar_matrix += torch.eye(covar_matrix.shape[0])
    if do_correlation:
        variances = torch.diag(covar_matrix)
        normalizer = torch.sqrt(variances.unsqueeze(0)*variances.unsqueeze(1))
        covar_matrix /= normalizer

    chosen_orig_idx = [ucb_sortidx[0].item()]
    chosen_idx = [0]
    while len(chosen_idx) < ack_batch_size:
        K_values = []
        for idx in range(preds.shape[1]):
            if idx in chosen_idx:
                K_values += [0]
                continue

            cur_chosen = torch.LongTensor(chosen_idx+[idx])
            logdet = torch.logdet(covar_matrix[cur_chosen, :][:, cur_chosen]).item()
            K_values += [logdet]

        K_values = np.array(K_values)

        factor = ucb.std()/K_values.std()
        K_values *= factor
        K_ucb = ucb + params.diversity_coeff*K_values
        sort_idx = K_ucb.argsort()

        for idx in sort_idx:
            idx = int(idx)
            if idx not in chosen_idx:
                break

        choice = ucb_sortidx[idx].item()
        chosen_idx += [idx]
        chosen_orig_idx += [choice]

    return list(chosen_orig_idx)


def acquire_batch_self_hsic(
    params,
    candidate_points_preds : torch.tensor, # (num_samples, num_candidate_points)
    skip_idx,
    mves_compute_batch_size,
    ack_batch_size,
    device : str = "cuda",
    true_labels=None,
    pred_weighting=0,
)-> torch.tensor:

    num_candidate_points = candidate_points_preds.shape[1]
    num_samples = candidate_points_preds.shape[0]
    min_pred = candidate_points_preds.min().to(device)
    ei = candidate_points_preds.mean(dim=0).to(device)
    ei = ei-ei.min()+0.1

    batch_idx = set()
    remaining_idx_set = set(range(num_candidate_points))
    remaining_idx_set = remaining_idx_set.difference(skip_idx)
    batch_dist_matrix = None # (num_samples, num_samples, 1)
    batch_sum_pred = 0

    while len(batch_idx) < ack_batch_size:
        if len(batch_idx) > 0 and len(batch_idx) % 10 == 0 and type(true_labels) != type(None):
            print(len(batch_idx), list(np.sort(true_labels[list(batch_idx)])[-5:]))

        best_idx = None
        best_idx_dist_matrix = None
        best_hsic = None

        num_remaining = len(remaining_idx_set)
        num_batches = num_remaining // mves_compute_batch_size + 1
        remaining_idx = torch.tensor(list(remaining_idx_set), device=device)

        for bi in range(num_batches):
            bs = bi*mves_compute_batch_size
            be = min((bi+1)*mves_compute_batch_size, num_remaining)
            cur_batch_size = be-bs
            if cur_batch_size == 0:
                continue
            idx = remaining_idx[bs:be]

            pred = candidate_points_preds[:, idx].to(device) # (num_samples, cur_batch_size)
            dist_matrix = hsic.sqdist(pred) # (num_samples, num_samples, cur_batch_size)

            assert list(dist_matrix.shape) == [
            num_samples,
            num_samples,
            cur_batch_size], str(dist_matrix.shape) + " == " \
                    + str([num_samples, num_samples, cur_batch_size])

            if batch_dist_matrix is not None:
                dist_matrix += batch_dist_matrix # (n, n, m)

            batch_kernel_matrix = getattr(hsic, params.hsic_kernel_fn)(dist_matrix) \
                    .detach() \
                    .permute([2, 0, 1]) \
                    .unsqueeze(-1) # (m, n, n, 1)

            assert list(batch_kernel_matrix.shape) == [
                    cur_batch_size,
                    num_samples,
                    num_samples,
                    1], str(batch_kernel_matrix.shape) + " == " \
                            + str([cur_batch_size, num_samples, num_samples, 2])

            self_hsic = torch.sqrt(
                    hsic.total_hsic_parallel(
                        batch_kernel_matrix.repeat([
                            1, 1, 1, 2])).detach()).view(-1)

            assert self_hsic.shape[0] == cur_batch_size
            assert not ops.isinf(self_hsic)
            assert not ops.isnan(self_hsic)

            del batch_kernel_matrix
            gc.collect()
            torch.cuda.empty_cache()

            if pred_weighting > 0:
                cur_batch_ei = (ei[bs:be] + batch_sum_pred) / (len(batch_idx) + 1)
                if pred_weighting == 1:
                    self_hsic *= cur_batch_ei
                elif pred_weighting == 2:
                    self_hsic = self_hsic * cur_batch_ei.std() + cur_batch_ei

            sorted_idx = self_hsic.cpu().numpy().argsort()
            best_cur_idx = sorted_idx[-1]

            if best_idx is None or best_hsic < self_hsic[best_cur_idx]:
                best_idx = idx[best_cur_idx].item()
                best_idx_dist_matrix = dist_matrix[:, :, best_cur_idx:best_cur_idx+1].detach()
                best_hsic = self_hsic[best_cur_idx].item()

        assert best_hsic is not None
        assert best_idx_dist_matrix is not None
        assert best_idx is not None

        batch_sum_pred += ei[best_idx]
        batch_dist_matrix = best_idx_dist_matrix
        best_hsic_overall = best_hsic
        remaining_idx_set.remove(best_idx)
        batch_idx.update({best_idx})

    return batch_idx, best_hsic



def acquire_batch_mves_sid(
    params,
    opt_values : torch.tensor, # (num_samples, num_opt_values)
    candidate_points_preds : torch.tensor, # (num_samples, num_candidate_points)
    skip_idx,
    mves_compute_batch_size,
    ack_batch_size,
    greedy_ordering=False,
    device : str = "cuda",
    true_labels=None,
    pred_weighting=False,
    normalize=True,
    divide_by_std=False,
    double=False,
    opt_weighting=None,
    min_hsic_increase=0.05,
)-> torch.tensor:

    opt_values = opt_values.to(device)

    num_candidate_points = candidate_points_preds.shape[1]
    num_samples = candidate_points_preds.shape[0]
    min_pred = candidate_points_preds.min().to(device)
    ei = candidate_points_preds.mean(dim=0).to(device)
    ei = ei-ei.min()+0.1

    if opt_values.ndimension() == 1:
        opt_values.unsqueeze(-1)
    assert opt_values.ndimension() == 2

    if opt_weighting is None:
        opt_dist_matrix = hsic.sqdist(opt_values.unsqueeze(1))
    else:
        assert len(opt_weighting.shape) == 1, opt_weighting.shape
        opt_dist_matrix = hsic.sqdist(opt_values.unsqueeze(1), collect=False) # (n, n, 1, k)
        opt_dist_matrix *= (opt_weighting**2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    opt_kernel_matrix = getattr(hsic, params.hsic_kernel_fn)(opt_dist_matrix).detach()  # shape (n=num_samples, n, 1)

    opt_normalizer = torch.log(hsic.total_hsic(opt_kernel_matrix.repeat([1, 1, 2])).detach()).view(-1)
    opt_normalizer_exp = torch.exp(0.5 * opt_normalizer)
    if ops.isnan(opt_normalizer_exp):
        return [], 0.0

    print('opt_self_hsic:', opt_normalizer_exp)
    assert list(opt_kernel_matrix.shape) == [num_samples, num_samples, 1], str(opt_kernel_matrix.shape) + " == " + str([num_samples, num_samples, 1])
    opt_kernel_matrix = opt_kernel_matrix.permute([2, 0, 1]).unsqueeze(-1) # (1, n, n, 1)

    batch_idx = []
    remaining_idx_set = set(range(num_candidate_points))
    remaining_idx_set = remaining_idx_set.difference(skip_idx)
    batch_dist_matrix = None # (num_samples, num_samples, 1)

    regular_batch_opt_kernel_matrix = opt_kernel_matrix.repeat([mves_compute_batch_size, 1, 1, 1]) # (mves_compute_batch_size, n, n, 1)

    best_hsic_vec = []
    while len(batch_idx) < ack_batch_size:
        #print("len(batch_idx):", len(batch_idx))
        if len(batch_idx) > 0 and len(batch_idx) % 10 == 0 and type(true_labels) != type(None):
            print(len(batch_idx), list(np.sort(true_labels[batch_idx])[-5:]))
        best_idx = None
        best_idx_dist_matrix = None
        best_hsic = None

        num_remaining = len(remaining_idx_set)
        num_batches = num_remaining // mves_compute_batch_size + 1
        remaining_idx = torch.tensor(list(remaining_idx_set), device=device)
        idx_hsic_values = []

        for bi in range(num_batches):
            bs = bi*mves_compute_batch_size
            be = min((bi+1)*mves_compute_batch_size, num_remaining)
            cur_batch_size = be-bs
            if cur_batch_size == 0:
                continue
            idx = remaining_idx[bs:be]

            pred = candidate_points_preds[:, idx].to(device) # (num_samples, cur_batch_size)
            dist_matrix = hsic.sqdist(pred) # (num_samples, num_samples, cur_batch_size) # not doing dimwise but doing hsic in || for multiple vars

            if pred_weighting == 1:
                predsqrt_matrix = pred-min_pred + 0.1
                predsqrt_matrix = torch.sqrt(predsqrt_matrix.unsqueeze(0))*torch.sqrt(predsqrt_matrix.unsqueeze(1))
                assert predsqrt_matrix.shape == dist_matrix.shape
                dist_matrix /= predsqrt_matrix
            elif pred_weighting == 2:
                ei_batch = ei[idx].unsqueeze(0).unsqueeze(0)
                assert ei_batch.shape[-1] == cur_batch_size
                dist_matrix *= (ei_batch**2)

            assert list(dist_matrix.shape) == [num_samples, num_samples, cur_batch_size], str(dist_matrix.shape) + " == " + str([num_samples, num_samples, cur_batch_size])

            if batch_dist_matrix is not None:
                dist_matrix += batch_dist_matrix # (n, n, m)

            batch_kernel_matrix = getattr(hsic, params.hsic_kernel_fn)(dist_matrix).detach().permute([2, 0, 1]).unsqueeze(-1) # (m, n, n, 1)
            assert list(batch_kernel_matrix.shape) == [cur_batch_size, num_samples, num_samples, 1], str(batch_kernel_matrix.shape) + " == " + str([cur_batch_size, num_samples, num_samples, 2])

            assert ops.tensor_all(batch_kernel_matrix <= 1.+1e-5), batch_kernel_matrix.max().item()

            if normalize:
                self_kernel_matrix = batch_kernel_matrix.repeat([1, 1, 1, 2]).detach() # (m, n, n, 2)
                assert list(self_kernel_matrix.shape) == [cur_batch_size, num_samples, num_samples, 2], str(self_kernel_matrix.shape) + " == " + str([cur_batch_size, num_samples, num_samples, 2])

                if double:
                    hsic_logvar = hsic.total_hsic_parallel(self_kernel_matrix.double()).detach()
                else:
                    hsic_logvar = hsic.total_hsic_parallel(self_kernel_matrix).detach()
                #assert ops.tensor_all(hsic_logvar > 0)
                hsic_logvar = torch.log(hsic_logvar).view(-1).float()
                assert hsic_logvar.shape[0] == cur_batch_size
                assert not ops.isinf(hsic_logvar)
                assert not ops.isnan(hsic_logvar)

            if cur_batch_size == mves_compute_batch_size:
                kernels = torch.cat([batch_kernel_matrix, regular_batch_opt_kernel_matrix], dim=-1)
            else:
                assert cur_batch_size < mves_compute_batch_size
                last_batch_opt_kernel_matrix = regular_batch_opt_kernel_matrix[:cur_batch_size]
                kernels = torch.cat([batch_kernel_matrix, last_batch_opt_kernel_matrix], dim=-1)

            assert list(kernels.shape) == [cur_batch_size, num_samples, num_samples, 2], str(kernels.shape)

            total_hsic = hsic.total_hsic_parallel(kernels)
            assert list(total_hsic.shape) == [cur_batch_size], str(total_hsic.shape)

            if normalize:
                if divide_by_std:
                    normalizer = torch.exp(hsic_logvar + 0.5 * opt_normalizer)
                else:
                    normalizer = torch.exp(0.5 * (hsic_logvar + opt_normalizer))
                assert total_hsic.shape[0] == normalizer.shape[0]
                total_hsic /= normalizer

            sorted_idx = total_hsic.cpu().numpy().argsort()
            del kernels
            gc.collect()
            torch.cuda.empty_cache()
            idx_hsic_values += [total_hsic.detach()]

            best_cur_idx = sorted_idx[-1]

            if best_idx is None or best_hsic < total_hsic[best_cur_idx]:
                best_idx = idx[best_cur_idx].item()
                best_idx_dist_matrix = dist_matrix[:, :, best_cur_idx:best_cur_idx+1].detach()
                best_hsic = total_hsic[best_cur_idx].item()

        if greedy_ordering:
            break

        assert best_hsic is not None
        assert best_idx_dist_matrix is not None
        assert best_idx is not None
        best_hsic_vec += [best_hsic]
        if len(best_hsic_vec) > 1 and best_hsic_vec[-1] < best_hsic_vec[-2]+min_hsic_increase:
            break

        batch_dist_matrix = best_idx_dist_matrix
        best_hsic_overall = best_hsic
        remaining_idx_set.remove(best_idx)
        assert best_idx not in batch_idx
        batch_idx += [best_idx]

    if greedy_ordering:
        idx_hsic_values = torch.cat(idx_hsic_values).cpu().numpy()
        batch_idx = set(idx_hsic_values.argsort()[-ack_batch_size:].tolist())
        best_hsic = idx_hsic_values[-1]

    print('best_hsic_vec', best_hsic_vec)
    return batch_idx, best_hsic


def acquire_batch_via_grad_er(
    params,
    model_ensemble: Callable[[torch.tensor], torch.tensor],
    input_shape: List[int],
    seed: torch.tensor = None,
    one_hot=True,
) -> torch.tensor:

    ack_batch_size = params.ack_batch_size
    if seed is None:
        input_tensor = torch.randn(
            [ack_batch_size] + input_shape, device=params.device, requires_grad=True
        )
    else:
        assert seed.shape[0] == ack_batch_size
        input_tensor = torch.tensor(seed, device=params.device, requires_grad=True)

    if one_hot:
        assert input_shape.ndimension() == 2
        input_tensor = torch.nn.functional.softmax(input_tensor, dim=-1)

    optim = torch.optim.Adam([input_tensor], lr=params.batch_opt_lr)
    kernel_fn = getattr(hsic, 'two_vec_' + params.hsic_kernel_fn)

    for step_iter in range(params.batch_opt_num_iter):
        preds = model_ensemble(input_tensor) # (ack_batch_size, num_samples)
        assert preds.ndimension() == 2
        preds = preds.transpose(0, 1)
        loss = -torch.mean(preds)

        optim.zero_grad()
        loss.backward()
        optim.step()

    return input_tensor.detach()


def optimize_model_input(
    params,
    input_shape,
    model_ensemble: Callable[[torch.tensor], torch.tensor],
    seed=None,
    hsic_diversity_lambda=0.,
    normalize_hsic=False,
    one_hot=True,
):
    if seed is None:
        input_tensor = torch.randn([num_points_to_optimize] + input_shape, device=params.device, requires_grad=True)
    else:
        assert seed.ndimension() == 2
        assert seed.shape[0] == num_points_to_optimize
        input_tensor = torch.tensor(seed, device=params.device, requires_grad=True)

    if one_hot:
        assert input_shape.ndimension() == 2
        input_tensor = torch.nn.functional.softmax(input_tensor, dim=-1)

    optim = torch.optim.Adam([input_tensor], lr=params.input_opt_lr)
    progress = tnrange(params.input_opt_num_iter)
    for step_iter in progress:
        preds = model_ensemble(input_tensor) # (num_samples, ack_batch_size)
        assert preds.ndimension() == 2

        loss = -torch.mean(preds)

        postfix = {'normal_loss' : loss.item()}
        kernel_fn = getattr(hsic, 'dimwise_' + params.hsic_kernel_fn)
        if hsic_diversity_lambda > 1e-9:
            kernels = kernel_fn(preds)
            total_hsic = hsic.total_hsic(kernels)

            if normalize_hsic:
                normalizer = hsic.total_hsic_parallel(kernels.permute(2, 0, 1).unsqueeze(-1).repeat([1, 1, 1, 2])).view(-1)
                normalizer = torch.exp(normalizer.log().sum()*0.5)
                total_hsic /= normalizer

            loss += hsic_diversity_lambda*total_hsic
            postfix['hsic_loss'] = total_hsic.item()
            postfix['hsic_loss_real'] = (hsic_diversity_lambda*total_hsic).item()
        else:
            kernels = kernel_fn(preds)
            total_hsic = hsic.total_hsic(kernels)
            postfix['hsic_loss'] = total_hsic.item()

        optim.zero_grad()
        loss.backward()
        optim.step()
        progress.set_postfix(postfix)

    return input_tensor.detach(), preds.detach()


def optimize_model_input_pdts(
    params,
    input_shape,
    model_ensemble: Callable[[torch.tensor], torch.tensor],
    num_points_to_optimize,
    seed=None,
    one_hot=True,
    input_transform=lambda x : x,
    hsic_diversity_lambda=0,
    normalize_hsic=True,
    jupyter=False,
):
    print("optimize_model_input_pdts, num_points_to_optimize:", num_points_to_optimize)
    assert num_points_to_optimize > 1
    if seed is None:
        input_tensor = torch.randn([num_points_to_optimize] + input_shape, device=params.device, requires_grad=True)
    else:
        assert seed.ndimension() == 2
        assert seed.shape[0] == num_points_to_optimize
        input_tensor = torch.tensor(seed, device=params.device, requires_grad=True)
    optim = torch.optim.Adam([input_tensor], lr=params.input_opt_lr)

    if jupyter:
        progress = tnrange(params.input_opt_num_iter)
    else:
        progress = trange(params.input_opt_num_iter)

    for step_iter in progress:
        if one_hot:
            assert len(input_shape) == 2
            input_tensor2 = torch.nn.functional.softmax(input_tensor, dim=-1)
        input_tensor2 = input_transform(input_tensor)
        if hsic_diversity_lambda > 1e-9:
            preds, _ = model_ensemble(input_tensor2) # (num_samples,)
            assert preds.ndimension() == 2
        else:
            preds, _ = model_ensemble(input_tensor2) # (num_samples,)
            #assert preds.ndimension() == 1
        #assert preds.shape[0] == num_points_to_optimize, str(preds.shape) + " == " + str(num_points_to_optimize)

        loss = -torch.mean(preds)
        mean_loss = loss.item()

        hsic_loss = 0
        if hsic_diversity_lambda > 1e-9:
            """
            kernel_fn = getattr(hsic, "two_vec_" + params.hsic_kernel_fn)
            kernels = kernel_fn(preds, preds)  # shape (n=num_samples, n, 1)
            total_hsic = torch.sqrt(hsic.total_hsic(kernels.repeat([1, 1, 2]))).view(-1)
            """

            kernel_fn = getattr(hsic, 'dimwise_' + params.hsic_kernel_fn)
            kernels = kernel_fn(preds).double()
            #kernels = kernel_fn(input_tensor.transpose(0, 1))
            total_hsic = hsic.total_hsic(kernels)
            #print(total_hsic.item())

            if normalize_hsic:
                normalizer = hsic.total_hsic_parallel(kernels.permute([2, 0, 1]).unsqueeze(-1).repeat([1, 1, 1, 2])).view(-1)
                #print(kernels[:, :, 6])
                #print(hsic.total_hsic(kernels[:, :, 6].unsqueeze(-1).repeat([1, 1, 2])).item())
                #print(normalizer)
                assert normalizer.shape[0] == num_points_to_optimize
                normalizer = normalizer.log().sum()/num_points_to_optimize
                total_hsic = torch.exp(total_hsic.log()-normalizer)
                #print('total_hsic2:', total_hsic.item())

            hsic_loss = total_hsic.item()
            loss += hsic_diversity_lambda*total_hsic.float()
            #postfix['hsic_loss'] = total_hsic.item()
            #postfix['hsic_loss_real'] = (hsic_diversity_lambda*total_hsic).item()

        optim.zero_grad()
        loss.backward()
        optim.step()
        if normalize_hsic and hsic_diversity_lambda > 1e-9:
            progress.set_description("%2.6f, %1.2f, %1.2f" % (mean_loss, hsic_loss, normalizer.item()))
            #progress.set_description("%2.6f, %1.2f" % (mean_loss, hsic_loss))
        else:
            progress.set_description("%2.6f, %1.2f" % (mean_loss, hsic_loss))

    with torch.no_grad():
        if one_hot:
            assert len(input_shape) == 2
            input_tensor2 = torch.nn.functional.softmax(input_tensor, dim=-1)
        input_tensor2 = input_transform(input_tensor)
        preds, _ = model_ensemble(input_tensor2) # (num_samples,)
        
    return input_tensor.detach(), preds.detach()


def two_dist_hsic(
        X : torch.tensor, # (num_samples, ack_batch_size)
        opt_values_kernel_matrix : torch.tensor,
        kernel_fn,
        do_mean=False
):
    assert X.ndimension() == 2
    num_samples = X.shape[0]
    ack_batch_size = X.shape[1]

    assert opt_values_kernel_matrix.shape[0] == num_samples
    assert opt_values_kernel_matrix.shape[1] == num_samples

    if opt_values_kernel_matrix.ndimension() == 2:
        opt_values_kernel_matrix = opt_values_kernel_matrix.unsqueeze(-1)
    assert opt_values_kernel_matrix.ndimension() == 3
    assert opt_values_kernel_matrix.shape[2] == 1, opt_values_kernel_matrix.shape

    new_batch_matrix = kernel_fn(X, X, do_mean=do_mean)  # return is of shape (n=num_samples, n, 1)
    kernels = torch.cat([new_batch_matrix, opt_values_kernel_matrix], dim=-1)

    return hsic.total_hsic(kernels)


def acquire_batch_via_grad_hsic(
    params,
    model_ensemble: Callable[[torch.tensor], torch.tensor],
    input_shape: List[int],
    opt_values: torch.tensor, # (num_samples, preds)
    ack_batch_size,
    device,
    hsic_condense_penalty,
    biased_hsic=False,
    do_mean=False,
    seed: torch.tensor = None,
    normalize_hsic=True,
    one_hot=True,
    input_transform=lambda x : x,
    sparse_hsic_penalty=0,
    sparse_hsic_threshold=0.,
    jupyter: bool = False,
) -> torch.tensor:

    assert sparse_hsic_threshold >= 0
    assert sparse_hsic_threshold < 1
    do_sparse_hsic = sparse_hsic_penalty > 1e-9

    print('acquire_batch_via_grad_hsic: ack_batch_size', ack_batch_size)
    if seed is None:
        input_tensor = torch.randn([ack_batch_size] + input_shape, device=device, requires_grad=True)
    else:
        assert seed.shape[0] == ack_batch_size
        input_tensor = torch.tensor(seed, device=device, requires_grad=True)

    if do_sparse_hsic:
        l1_tensor = torch.ones(ack_batch_size, device=device, requires_grad=True)
        l1_tensor_zero = torch.zeros(ack_batch_size, device=device)
        optim = torch.optim.Adam([input_tensor, l1_tensor], lr=params.hsic_opt_lr)
    else:
        optim = torch.optim.Adam([input_tensor], lr=params.hsic_opt_lr)

    kernel_fn = getattr(hsic, "two_vec_" + params.hsic_kernel_fn)
    opt_kernel_matrix = kernel_fn(opt_values, opt_values, do_mean=do_mean)[:, :, 0]  # shape (n=num_samples, n, 1)

    if jupyter:
        progress = tnrange(params.hsic_opt_num_iter)
    else:
        progress = trange(params.hsic_opt_num_iter)

    for step_iter in progress:
        if one_hot:
            assert len(input_shape) == 2
            input_tensor2 = torch.nn.functional.softmax(input_tensor, dim=-1)
        input_tensor2 = input_transform(input_tensor)

        preds, _ = model_ensemble(input_tensor2) # (num_samples, ack_batch_size)
        if do_sparse_hsic:
            preds *= l1_tensor.unsqueeze(0)
        assert preds.ndimension() == 2
        assert opt_kernel_matrix.shape[0] == preds.shape[0], str(opt_kernel_matrix.shape) + "[0] == " + str(preds.shape) + "[0]"

        mean_pred = torch.mean(preds)
        loss = -hsic_condense_penalty[0]*mean_pred
        preds_kernel_matrix = kernel_fn(preds, preds, do_mean=do_mean)[:, :, 0]
        
        total_hsic = two_dist_hsic(preds, opt_kernel_matrix, kernel_fn, do_mean)
        hsic_xy = hsic.hsic_xy(opt_kernel_matrix, preds_kernel_matrix, biased=biased_hsic, normalized=normalize_hsic)

        hsic_loss = hsic_xy.item()
        loss += -hsic_condense_penalty[1]*hsic_xy
        if do_sparse_hsic:
            l1_loss = torch.nn.functional.l1_loss(l1_tensor, l1_tensor_zero)
            loss += sparse_hsic_penalty*l1_loss
            progress.set_description("%1.5f, %1.5f, %d" % (hsic_loss, l1_loss.item(), (l1_tensor > sparse_hsic_threshold).sum()))
        else:
            progress.set_description("%1.5f, %1.4f" % (mean_pred, hsic_loss))

        optim.zero_grad()
        loss.backward()
        optim.step()

    if do_sparse_hsic:
        return input_tensor.detach()[l1_tensor > sparse_hsic_threshold], hsic_loss
    else:
        return input_tensor.detach(), hsic_loss


def acquire_batch_via_grad_hsic2(
    params,
    model_ensemble: Callable[[torch.tensor], torch.tensor],
    input_shape: List[int],
    opt_values: torch.tensor, # (num_samples, preds)
    ack_batch_size,
    device,
    hsic_condense_penalty,
    biased_hsic=False,
    do_mean=False,
    seed: torch.tensor = None,
    normalize_hsic=True,
    one_hot=True,
    input_transform=lambda x : x,
    jupyter: bool = False,
) -> torch.tensor:

    print('acquire_batch_via_grad_hsic: ack_batch_size', ack_batch_size)
    input_tensor_list = []
    complete_input_tensor = torch.randn([ack_batch_size] + input_shape, device=device, requires_grad=False)

    kernel_fn = getattr(hsic, "two_vec_" + params.hsic_kernel_fn)
    opt_kernel_matrix = kernel_fn(opt_values, opt_values, do_mean=do_mean)[:, :, 0]  # shape (n=num_samples, n, 1)

    hsic_loss_so_far = 0
    hsic_loss = 0
    if jupyter:
        batch_progress = tnrange(ack_batch_size)
    else:
        batch_progress = trange(ack_batch_size)
    for ack_iter in batch_progress:
        input_tensor = torch.zeros([ack_iter+1] + input_shape, device=device, requires_grad=True)
        with torch.no_grad():
            input_tensor[:ack_iter] = complete_input_tensor[:ack_iter]
        optim = torch.optim.Adam([input_tensor], lr=params.hsic_opt_lr)
        for step_iter in range(params.hsic_opt_num_iter):
            if one_hot:
                assert len(input_shape) == 2
                input_tensor2 = torch.nn.functional.softmax(input_tensor, dim=-1)
            input_tensor2 = input_transform(input_tensor)
            #print(input_tensor2)
            assert not ops.is_nan(input_tensor2), str(ack_iter) + ", " + str(step_iter)

            preds, _ = model_ensemble(input_tensor2) # (num_samples, ack_batch_size)
            assert preds.ndimension() == 2
            assert opt_kernel_matrix.shape[0] == preds.shape[0], str(opt_kernel_matrix.shape) + "[0] == " + str(preds.shape) + "[0]"

            mean_pred = torch.mean(preds)
            loss = -hsic_condense_penalty[0]*mean_pred
            preds_kernel_matrix = kernel_fn(preds, preds, do_mean=do_mean)[:, :, 0]

            hsic_xy = hsic.hsic_xy(opt_kernel_matrix, preds_kernel_matrix, biased=biased_hsic, normalized=normalize_hsic)

            # if preds end up having 0 stddev hsic estimator becoming infinity
            if not ops.is_finite(hsic_xy):
                break

            hsic_loss = hsic_xy.item()
            loss += -hsic_condense_penalty[1]*hsic_xy
            #print('loss', loss.item(), hsic_loss, mean_pred.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

        batch_progress.set_description("%1.5f, %1.5f" % (hsic_loss, mean_pred.item()))
        if hsic_loss <= hsic_loss_so_far + 0.05:
            break
        hsic_loss_so_far = hsic_loss
        complete_input_tensor[:ack_iter+1, :, :] = input_tensor.detach()

    return complete_input_tensor[:ack_iter].detach(), hsic_loss_so_far


def acquire_batch_via_grad_opt_location(
    params,
    model_ensemble: Callable[[torch.tensor], torch.tensor],
    input_shape: List[int],
    opt_locations: torch.tensor, # (num_samples, num_features)
    ack_batch_size,
    device,
    hsic_condense_penalty,
    biased_hsic=False,
    do_mean=False,
    seed: torch.tensor = None,
    normalize_hsic=True,
    one_hot=True,
    input_transform=lambda x : x,
    jupyter: bool = False,
) -> torch.tensor:

    print('acquire_batch_via_grad_hsic: ack_batch_size', ack_batch_size)
    input_tensor_list = []
    complete_input_tensor = torch.randn([ack_batch_size] + input_shape, device=device, requires_grad=False)

    kernel_fn = getattr(hsic, "two_vec_" + params.hsic_kernel_fn)
    opt_kernel_matrix = kernel_fn(opt_locations, opt_locations, do_mean=do_mean)[:, :, 0]  # shape (n=num_samples, n, 1)

    hsic_loss_so_far = 0
    if jupyter:
        batch_progress = tnrange(ack_batch_size)
    else:
        batch_progress = trange(ack_batch_size)
    for ack_iter in batch_progress:
        input_tensor = torch.zeros([ack_iter+1] + input_shape, device=device, requires_grad=True)
        with torch.no_grad():
            input_tensor[:ack_iter] = complete_input_tensor[:ack_iter]
        optim = torch.optim.Adam([input_tensor], lr=params.hsic_opt_lr)
        for step_iter in range(params.hsic_opt_num_iter):
            if one_hot:
                assert len(input_shape) == 2
                input_tensor2 = torch.nn.functional.softmax(input_tensor, dim=-1)
            input_tensor2 = input_transform(input_tensor)

            preds, _ = model_ensemble(input_tensor2) # (num_samples, ack_batch_size)
            assert preds.ndimension() == 2
            assert opt_kernel_matrix.shape[0] == preds.shape[0], str(opt_kernel_matrix.shape) + "[0] == " + str(preds.shape) + "[0]"

            mean_pred = torch.mean(preds)
            loss = -hsic_condense_penalty[0]*mean_pred

            preds_kernel_matrix = kernel_fn(preds, preds, do_mean=do_mean)[:, :, 0]
            hsic_xy = hsic.hsic_xy(opt_kernel_matrix, preds_kernel_matrix, biased=biased_hsic, normalized=normalize_hsic)

            # if preds end up having 0 stddev hsic estimator becoming infinity
            if not ops.is_finite(hsic_xy):
                break

            hsic_loss = hsic_xy.item()
            loss += -hsic_condense_penalty[1]*hsic_xy

            optim.zero_grad()
            loss.backward()
            optim.step()
        batch_progress.set_description("%1.5f, %1.5f" % (mean_pred.item(), hsic_loss))
        if hsic_loss <= hsic_loss_so_far + 0.05:
            break
        hsic_loss_so_far = hsic_loss
        complete_input_tensor[:ack_iter+1, :, :] = input_tensor.detach()

    return complete_input_tensor[:ack_iter].detach(), hsic_loss_so_far


def acquire_batch_pi(
    model,
    inputs,
    sampled_labels,
    sampled_idx: Set[int],
    batch_size: int,
    exp: Optional = None,
) -> List[int]:
    n_preds = 1000
    max_val = sampled_labels.max()
    bnn_model, guide = model

    batch = []
    acquirable_idx = list(set(range(len(inputs))).difference(sampled_idx))
    inputs = inputs[acquirable_idx]

    with torch.no_grad():
        preds = torch.stack([guide()(inputs).squeeze() for _ in range(n_preds)])

    prob_improvement = (preds > max_val).sum(dim=0).float() / n_preds

    batch = prob_improvement.sort(descending=True)[1][:batch_size].tolist()
    batch = [acquirable_idx[i] for i in batch]

    assert len(batch) == batch_size or not acquirable_idx
    return batch


def optimize_inputs(
    inputs,
    input_optimizer,
    model,
    n_steps: int,
    constrain_every: Optional[int] = None,
    bounds=None,
):
    constrain_every = np.inf if constrain_every is None else constrain_every

    def _optimize_inputs():
        if bounds is not None and step % constrain_every == 0:
            for i in range(len(bounds)):
                inputs.data[:, i].clamp_(*bounds[i])

        input_optimizer.zero_grad()
        output = model(inputs)
        # don't know if we should need `retain_graph`; there might
        # be something better I should be doing...
        (-output).backward(torch.ones_like(output), retain_graph=True)

    for step in range(n_steps):
        input_optimizer.step(_optimize_inputs())

    if bounds is not None:
        for i in range(len(bounds)):
            inputs.data[:, i].clamp_(*bounds[i])

def get_pdts_idx(preds, ack_batch_size, density=False):
    pdts_idx = set()

    sorted_preds_idx = []
    for i in range(preds.shape[0]):
        sorted_preds_idx += [np.argsort(preds[i].numpy())]
    sorted_preds_idx = np.array(sorted_preds_idx)

    if density:
        counts = np.zeros(sorted_preds_idx.shape[1])
        for rank in range(sorted_preds_idx.shape[1]):
            counts[:] = 0
            for idx in sorted_preds_idx[:, rank]:
                counts[idx] += 1
            counts_idx = counts.argsort()[::-1]
            j = 0
            while len(pdts_idx) < ack_batch_size and j < counts_idx.shape[0] and counts[counts_idx[j]] > 0:
                idx = int(counts_idx[j])
                pdts_idx.update({idx})
                j += 1
            if len(pdts_idx) >= ack_batch_size:
                break
    else:
        for i_model in range(sorted_preds_idx.shape[0]):
            for idx in sorted_preds_idx[i_model]:
                idx2 = int(idx)
                if idx2 not in pdts_idx:
                    pdts_idx.update({idx2})
                    break
            if len(pdts_idx) >= ack_batch_size:
                break

    pdts_idx = list(pdts_idx)
    return pdts_idx


def get_nll(
    preds,
    std,
    Y,
    output_noise_dist_fn,
    single_gaussian=True,
):
    assert len(preds.shape) == 2
    assert preds.shape == std.shape
    assert preds.shape[1] == Y.shape[0]
    m = preds.shape[0]
    if single_gaussian:
        output_dist = output_noise_dist_fn(
                preds.mean(0),
                std.mean(0))
        log_prob = -output_dist.log_prob(Y)
    else:
        output_dist = output_noise_dist_fn(
                preds.view(-1),
                std.view(-1))
        log_prob = -output_dist.log_prob(Y.repeat([m]))
    return log_prob


def get_pred_stats(
    pred_means,
    pred_std,
    Y,
    output_noise_dist_fn,
    sigmoid_coeff,
    train_Y=None,
):
    assert pred_means.shape[1] == Y.shape[0], "%s[1] == %s[0]" % (pred_means.shape, Y.shape)
    assert pred_means.shape == pred_std.shape,  "%s == %s" % (pred_means.shape, pred_std.shape)

    baseline_rmse = None
    if train_Y is not None:
        if sigmoid_coeff > 0:
            Y = utils.sigmoid_standardization(
                    Y,
                    train_Y.mean(),
                    train_Y.std(),
                    exp=torch.exp)
            train_Y = utils.sigmoid_standardization(
                    train_Y,
                    train_Y.mean(),
                    train_Y.std(),
                    exp=torch.exp)
        else:
            Y = utils.normal_standardization(
                    Y,
                    train_Y.mean(),
                    train_Y.std(),
                    exp=torch.exp)
            train_Y = utils.normal_standardization(
                    train_Y,
                    train_Y.mean(),
                    train_Y.std(),
                    exp=torch.exp)

        baseline_rmse = torch.sqrt(torch.mean((Y-train_Y.mean())**2)).item()

    log_prob = torch.mean(
                get_nll(
                    pred_means, 
                    pred_std, 
                    Y,
                    output_noise_dist_fn,
                    )
                ).item()

    pred_means_mean = pred_means.mean(dim=0)
    mse = (pred_means_mean-Y)**2
    rmse = torch.sqrt(torch.mean(mse)).item()
    
    pred_mean_std = pred_means.std(dim=0)
    std = pred_mean_std.mean().item()

    if pred_means_mean.shape[0] > 1:
        kt_corr = kendalltau(pred_means_mean, Y)[0]
        rmse_std_corr = float(pearsonr(torch.sqrt(mse).cpu().numpy(), pred_mean_std.cpu().numpy())[0])
    else:
        kt_corr = 0
        rmse_std_corr =0 


    ret = {
            'log_prob' : log_prob,
            'rmse' : rmse,
            'kt_corr' : kt_corr,
            'std' : std,
            'rmse_std_corr' : rmse_std_corr,
            'baseline_rmse' : baseline_rmse,
            }

    return ret


def get_ind_top_ood_pred_stats(
        preds, 
        std, 
        unscaled_Y,
        train_Y,
        output_noise_dist_fn, 
        idx_to_exclude,
        sigmoid_coeff,
        single_gaussian=True,
        num_to_eval=0,
):
    assert preds.shape[1] == unscaled_Y.shape[0], "%s[1] == %s[0]" % (preds.shape, unscaled_Y.shape)
    assert preds.shape == std.shape, "%s == %s" % (preds.shape, std.shape)
    assert unscaled_Y.shape[0] > 0

    if sigmoid_coeff > 0:
        Y = utils.sigmoid_standardization(
                unscaled_Y,
                train_Y.mean(),
                train_Y.std(),
                exp=torch.exp)
    else:
        Y = utils.normal_standardization(
                unscaled_Y,
                train_Y.mean(),
                train_Y.std(),
                exp=torch.exp)


    test_idx = list({i for i in range(Y.shape[0])}.difference(idx_to_exclude))
    if num_to_eval > 0:
        test_idx = test_idx[:num_to_eval]

    assert len(test_idx) > 0
    preds = preds[:, test_idx]
    std = std[:, test_idx]
    Y = Y[test_idx]
    unscaled_Y = unscaled_Y[test_idx]

    log_prob_list = []
    rmse_list = []
    kt_corr_list = []
    std_list = []
    rmse_std_corr_list = []
    pred_corr = []

    stats = {
            'log_prob' : [],
            'rmse' : [],
            'kt_corr' : [],
            'std' : [],
            'rmse_std_corr' : []
            }

    for frac in [1., 0.1, 0.01]:
        labels_sort_idx = torch.sort(unscaled_Y, descending=True)[1].cpu().numpy()
        n = max(int(labels_sort_idx.shape[0] * frac), 2)
        m = preds.shape[0]
        labels_sort_idx = labels_sort_idx[:n]
        labels2 = Y[labels_sort_idx]

        rand_idx = torch.randperm(n)[:200]
        rand_idx = labels_sort_idx[rand_idx]
        corr = np.corrcoef(preds[:, rand_idx].transpose(0, 1))
        pred_corr += [corr[np.tril_indices(corr.shape[0], k=-1)].mean()]

        ret = get_pred_stats(
                preds[:, labels_sort_idx],
                std[:, labels_sort_idx],
                labels2,
                output_noise_dist_fn,
                sigmoid_coeff,
                )

        for stat in stats:
            stats[stat] += [ret[stat]]

    pred_corr += [0]
    max_idx = [labels_sort_idx[0]]
    ret = get_pred_stats(
            preds[:, max_idx],
            std[:, max_idx],
            Y[max_idx],
            output_noise_dist_fn,
            sigmoid_coeff,
            )
    for stat in stats:
        stats[stat] += [ret[stat]]

    for stat in stats:
        print(stat, ":", stats[stat])

    return stats

def compute_ir_regret_ensemble(
        model_ensemble,
        X,
        Y,
        ack_all,
        ack_batch_size,
):
    preds, _ = model_ensemble(X) # (num_candidate_points, num_samples)
    preds = preds.detach()
    ei = preds.mean(dim=0).view(-1).cpu().numpy()
    ei_sortidx = np.argsort(ei)[-50:]
    ack = list(ack_all.union(list(ei_sortidx)))
    best_10 = np.sort(Y[ack])[-10:]
    best_batch_size = np.sort(Y[ack])[-ack_batch_size:]

    s = [best_10.mean(), best_10.max(), best_batch_size.mean(), best_batch_size.max()]

    return s, ei, ei_sortidx

def compute_idx_frac(ack_all, top_frac_idx):
    idx_frac = [len(ack_all.intersection(k))/len(k) for k in top_frac_idx]
    return idx_frac


def compute_std_ratio_matrix(old_std, preds, only_tril=True):
    new_std = preds.std(dim=0)
    std_ratio = old_std/new_std
    std_ratio = std_ratio
    std_ratio_matrix = std_ratio.unsqueeze(-1)/std_ratio.unsqueeze(0)
    std_ratio_matrix = torch.min(std_ratio_matrix, std_ratio_matrix.transpose(0, 1))
    std_ratio_matrix = std_ratio_matrix.detach().cpu().numpy()
    if only_tril:
        std_ratio_matrix = std_ratio_matrix[np.tril_indices(std_ratio_matrix.shape[0], k=-1)]
    return std_ratio_matrix


def pairwise_hsic(
    params, 
    preds, # (num_samples, num_points)
):
    with torch.no_grad():
        n = preds.shape[1]
        hsic_matrix = torch.zeros((n, n), device=params.device)
        kernel_fn = getattr(hsic, "dimwise_" + params.hsic_kernel_fn)
        kernels = kernel_fn(preds)
        for i in range(n):
            for j in range(i+1, n):
                hsic_matrix[i, j] = hsic.hsic_xy(
                        kernels[:, :, i], 
                        kernels[:, :, j],
                        normalized=params.normalize_hsic,
                        ).item()
                hsic_matrix[j, i] = hsic_matrix[i, j]
    return hsic_matrix


def hsic_with_batch(
    params,
    batch_preds, # (num_samples, num_points)
    other_preds, # (num_samples, num_points)
    hsic_custom_kernel=None,
):
    with torch.no_grad():

        if hsic_custom_kernel is not None:
            batch_preds = hsic_custom_kernel(batch_preds.transpose(0, 1)).transpose(0, 1)
            other_preds = hsic_custom_kernel(other_preds.transpose(0, 1)).transpose(0, 1)

        N = other_preds.shape[1]
        kernel_fn = getattr(hsic, 'two_vec_' + params.hsic_kernel_fn)
        batch_kernels = kernel_fn(batch_preds)[:, :, 0]
        kernel_fn = getattr(hsic, 'dimwise_' + params.hsic_kernel_fn)
        
        batch_size = 1000
        num_batches = int(math.ceil(float(N)/batch_size))
        batches = [min(i*batch_size, N) for i in range(num_batches+1)]

        hsic_vec = torch.zeros(N)

        for bi in range(len(batches)-1):
            bs = batches[bi]
            be = batches[bi+1]
            assert be > bs

            pred_kernels = kernel_fn(other_preds[:, bs:be])
            for i in range(be-bs):
                hsic_vec[bs+i] = hsic.hsic_xy(batch_kernels, pred_kernels[:, :, i], normalized=params.normalize_hsic).item()
                assert ops.isfinite(hsic_vec[bs+i]), str(hsic_vec[bs+i]) + "\t" + str(i)
        return hsic_vec


def max_batch_hsic(
    params,
    batch_preds, # (num_samples, num_points)
    other_preds, # (num_samples, num_points)
):
    num_points = other_preds.shape[1]
    kernel_fn = getattr(hsic, 'dimwise_' + params.hsic_kernel_fn)
    batch_kernels = kernel_fn(batch_preds)
    pred_kernels = kernel_fn(other_preds)

    hsic_vec = torch.zeros(num_points)
    for i in range(other_preds.shape[1]):
        max_hsic = 0
        for batch_i in range(batch_preds.shape[1]):
            hsic_val = hsic.hsic_xy(batch_kernels[:, :, batch_i], pred_kernels[:, :, i], normalized=params.normalize_hsic).item()
            if hsic_val > max_hsic:
                max_hsic = hsic_val
        hsic_vec[i] = max_hsic
    return hsic_vec


def pairwise_mi(
    params, 
    preds, # (num_samples, num_points)
):
    preds = preds.cpu().numpy()
    with torch.no_grad():
        n = preds.shape[1]
        mi_matrix = torch.zeros((n, n), device=params.device)
        for i in range(n):
            for j in range(i+1, n):
                mi_matrix[i, j] = float(estimate_mi(np.vstack([preds[:, i], preds[:, j]])))
                mi_matrix[j, i] = mi_matrix[i, j]
    return mi_matrix


def get_empirical_kriging_believer_ack(
    params,
    model,
    data,
    ack_batch_size,
    skip_idx,
    train_fn,
    cur_rmse,
):
    skip_idx = list(skip_idx)
    train_X, train_Y, X, Y = data
    ack_idx = []

    model = copy.deepcopy(model)
    optim = torch.optim.Adam(list(model.parameters()), lr=params.re_train_lr, weight_decay=params.re_train_l2)
    for batch_iter in range(ack_batch_size):
        with torch.no_grad():
            pred_means, pred_vars = model(X)
            er = pred_means.mean(dim=0).view(-1)
            std = pred_means.std(dim=0).view(-1)
            ucb_measure = er + params.ucb*std
            ucb_measure[skip_idx + ack_idx] = ucb_measure.min()
            ucb_sort, ucb_sortidx = torch.sort(ucb_measure, descending=True)

            ack_idx_to_condense = []
            #condense_size = params.num_diversity*ack_batch_size
            condense_size = params.num_diversity
            for i in range(ucb_sortidx.shape[0]):
                idx = ucb_sortidx[i].item()
                if idx in skip_idx:
                    continue
                ack_idx_to_condense += [idx]
                if len(ack_idx_to_condense) == condense_size:
                    break

        cur_ack_idx = get_empirical_condensation_ack3(
                params,
                X,
                Y,
                pred_means,
                pred_vars,
                model,
                params.re_train_lr,
                params.re_train_l2,
                ack_idx_to_condense,
                skip_idx,
                ack_batch_size,
                cur_rmse,
                idx_to_monitor=None,
                er_values=ucb_measure,
                seen_batch_size=params.re_train_batch_size,
                stat_fn=lambda preds, info : preds.std(dim=0),
                val_nll_metric=False,
                )

        ack_idx += [cur_ack_idx[0]]

        if len(ack_idx) >= ack_batch_size:
            break

        if params.sigmoid_coeff > 0:
            bY = utils.sigmoid_standardization(
                    train_Y,
                    train_Y.mean(),
                    train_Y.std(),
                    exp=torch.exp)
        else:
            bY = utils.normal_standardization(
                    train_Y,
                    train_Y.mean(),
                    train_Y.std(),
                    exp=torch.exp)

        bX = torch.cat([train_X, X[ack_idx]], dim=0)
        bY = torch.cat([bY, er[ack_idx]], dim=0)

        print('batch_iter', batch_iter)
        num_iter = 20
        for i in range(num_iter):
            means, variances = model(bX)
            nll = model.compute_negative_log_likelihood(
                bY,
                means,
                variances,
                return_mse=False,
            )
            optim.zero_grad()
            nll.backward()
            optim.step()

    return ack_idx


def get_bagging_er(
    params,
    model,
    data,
    ack_batch_size,
    skip_idx,
):
    train_X, train_Y, X, Y = data
    skip_idx = list(skip_idx)
    cur_ack_idx = []
    for ack_batch_iter in range(ack_batch_size):
        pred_means, _ = model.bagging_forward(X, num_to_bag=4)
        er = pred_means.mean(dim=0).view(-1)
        std = pred_means.std(dim=0).view(-1)
        ucb_measure = er + params.ucb*std
        ucb_measure[skip_idx + cur_ack_idx] = ucb_measure.min()

        max_idx = torch.argmax(ucb_measure)[0].item()
        cur_ack_idx += [max_idx]

    return cur_ack_idx


def get_nb_mcts_ack(
    params,
    orig_model,
    data,
    ack_batch_size,
    skip_idx,
    train_fn,
):
    pass


def get_kriging_believer_ack(
    params,
    orig_model,
    data,
    ack_batch_size,
    skip_idx,
    train_fn,
    parallel=False,
):
    skip_idx = list(skip_idx)
    train_X, train_Y, X, Y = data
    ack_idx = []
    fake_ack_idx = []

    ucb_measure = None
    if not parallel:
        model = copy.deepcopy(orig_model)
        optim = torch.optim.Adam(list(model.parameters()), lr=params.re_train_lr, weight_decay=params.re_train_l2)
    for batch_iter in range(ack_batch_size):
        if parallel:
            model = copy.deepcopy(orig_model)
            optim = torch.optim.Adam(list(model.parameters()), lr=params.re_train_lr, weight_decay=params.re_train_l2)
        with torch.no_grad():
            if ucb_measure is None:
                pred_means, pred_vars = model(X)
                er = pred_means.mean(dim=0).view(-1)
                std = pred_means.std(dim=0).view(-1)
                std_measure = std
                ucb_measure = er + params.ucb*std
                ucb_measure[skip_idx + ack_idx] = ucb_measure.min()
                ucb_sort_idx = torch.sort(ucb_measure, descending=True)[1]

            temp = list(set(ack_idx + fake_ack_idx))
            std_measure[skip_idx + fake_ack_idx] = -2
            std_measure_sort_idx = torch.sort(std_measure, descending=True)[1]
            num_fake_ack_candidates = X.shape[0] - (len(skip_idx) + len(fake_ack_idx))

            max_idx = ucb_sort_idx[0].item()
            ack_idx += [max_idx]
            num_candidates = X.shape[0] - (len(skip_idx) + len(ack_idx))

        if len(ack_idx) >= ack_batch_size:
            break

        if params.sigmoid_coeff > 0:
            bY = utils.sigmoid_standardization(
                    train_Y,
                    train_Y.mean(),
                    train_Y.std(),
                    exp=torch.exp)
        else:
            bY = utils.normal_standardization(
                    train_Y,
                    train_Y.mean(),
                    train_Y.std(),
                    exp=torch.exp)

        #fake_ack_idx = ack_idx
        #fake_ack_idx += [ucb_sort_idx[random.randint(1, num_candidates-1)].item()]
        #fake_ack_idx += [std_measure_sort_idx[0].item()]

        candidate_fake_ack_idx = std_measure_sort_idx[np.random.choice(num_fake_ack_candidates, size=10, replace=False)].cpu().numpy().tolist()
        if len(fake_ack_idx) > 0:
            corr_matrix = pred_means[:, fake_ack_idx + candidate_fake_ack_idx].transpose(0, 1)
            corr_matrix = torch.abs(ops.corrcoef(corr_matrix))
            corr = corr_matrix[len(fake_ack_idx):, :][:, :len(fake_ack_idx)].mean(dim=1)
            #print(corr_matrix.shape, corr.shape, len(fake_ack_idx), len(candidate_fake_ack_idx))
            corr_min_idx = torch.argmin(corr).item()
            print('corr_mean, corr_min', torch.mean(corr).item(), corr[corr_min_idx].item(), corr_min_idx)
            fake_ack_idx += [candidate_fake_ack_idx[corr_min_idx]]
        else:
            fake_ack_idx += [candidate_fake_ack_idx[0]]

        bX = torch.cat([train_X, X[fake_ack_idx]], dim=0)
        bY = torch.cat([bY, er[fake_ack_idx]], dim=0)

        print('batch_iter', batch_iter)
        num_iter = 20
        for i in range(num_iter):
            means, variances = model(bX)
            nll = model.compute_negative_log_likelihood(
                bY,
                means,
                variances,
                return_mse=False,
            )
            optim.zero_grad()
            nll.backward()
            optim.step()

        with torch.no_grad():
            pred_means, pred_vars = model(X)
            er = pred_means.mean(dim=0).view(-1)
            std = pred_means.std(dim=0).view(-1)
            std_measure = std
            ucb_measure = er + params.ucb*std
            ucb_measure[skip_idx + ack_idx] = ucb_measure.min()
            ucb_sort_idx = torch.sort(ucb_measure, descending=True)[1]

        """
        logging, _, _, _ = train_fn(
                params,
                params.re_train_batch_size,
                params.re_train_num_epochs,
                [train2_X, train2_Y, X, Y],
                model,
                optim,
                params.final_train_choose_type,
                )
        """

    return ack_idx


def get_noninfo_ack(
    params,
    ack_fun, 
    preds, 
    ack_batch_size,
    skip_idx,
    ack_iter_info=dict(),
):
    with torch.no_grad():
        er = preds.mean(dim=0).view(-1).cpu().numpy()
        std = preds.std(dim=0).view(-1).cpu().numpy()

        if "var" in ack_fun:
            er_sortidx = np.argsort(er/std)
        elif "ucb" in ack_fun:
            assert 'ucb_beta' in ack_iter_info
            er_sortidx = np.argsort(er + ack_iter_info['ucb_beta']*std)
        else:
            assert False, ack_fun  + " not implemented"

    if "none" in ack_fun:
        assert params.num_diversity >= 1
        cur_ack_idx = []
        num_to_select = int(math.ceil(ack_batch_size * params.num_diversity))
        for idx in er_sortidx[::-1]:
            if idx not in skip_idx:
                cur_ack_idx += [idx]
            if len(cur_ack_idx) >= num_to_select:
                break
        if params.num_diversity > 1:
            temp = utils.randint(num_to_select, ack_batch_size)
            cur_ack_idx = np.array(cur_ack_idx)[temp].tolist()
        assert len(cur_ack_idx) == ack_batch_size
    elif "hsic" in ack_fun:
        num_to_select = int(math.ceil(ack_batch_size * params.num_diversity))
        cur_ack_idx = er_diversity_selection_hsic(
                params, 
                preds, 
                skip_idx, 
                num_er=num_to_select,
                device=params.device)
    elif "detk" in ack_fun:
        num_to_select = int(math.ceil(ack_batch_size * params.num_diversity))
        cur_ack_idx = er_diversity_selection_detk(
                params, 
                preds, 
                skip_idx, 
                num_ucb=num_to_select,
                device=params.device)
    elif "pdts" in ack_fun:
        cur_ack_idx = set()
        er_sortidx = np.argsort(er)
        sorted_preds_idx = []
        for i in range(preds.shape[0]):
            sorted_preds_idx += [np.argsort(preds[i].numpy())]
        sorted_preds_idx = np.array(sorted_preds_idx)
        if "density" in ack_fun:
            counts = np.zeros(sorted_preds_idx.shape[1])
            for rank in range(sorted_preds_idx.shape[1]):
                counts[:] = 0
                for idx in sorted_preds_idx[:, rank]:
                    counts[idx] += 1
                counts_idx = counts.argsort()[::-1]
                j = 0
                while len(cur_ack_idx) < ack_batch_size and j < counts_idx.shape[0] and counts[counts_idx[j]] > 0:
                    idx = int(counts_idx[j])
                    cur_ack_idx.update({idx})
                    j += 1
                if len(cur_ack_idx) >= ack_batch_size:
                    break
        else:
            assert ack_fun == "pdts", ack_fun
            for i_model in range(sorted_preds_idx.shape[0]):
                for idx in sorted_preds_idx[i_model]:
                    idx2 = int(idx)
                    if idx2 not in cur_ack_idx:
                        cur_ack_idx.update({idx2})
                        break
                if len(cur_ack_idx) >= ack_batch_size:
                    break
        cur_ack_idx = list(cur_ack_idx)
    else:
        assert False, "Not implemented " + ack_fun
    assert len(cur_ack_idx) == ack_batch_size, len(cur_ack_idx)

    return cur_ack_idx


def get_info_ack(
    params, 
    preds, 
    ack_batch_size,
    skip_idx,
    labels,
):
    with torch.no_grad():
        er = preds.mean(dim=0).view(-1).cpu().numpy()
        std = preds.std(dim=0).view(-1).cpu().numpy()

        er += params.ucb*std

        if not params.compare_w_old:
            er[list(skip_idx)] = er.min()
        
        top_k = params.num_diversity

        sorted_preds_idx = torch.sort(preds)[1].cpu().numpy()
        #f.write('diversity\t' + str(len(set(sorted_preds_idx[:, -top_k:].flatten()))) + "\n")
        sorted_preds = torch.sort(preds, dim=1)[0]    
        #best_pred = preds.max(dim=1)[0].view(-1)
        #best_pred = sorted_preds[:, -top_k:]

        opt_weighting = None
        if params.measure == 'er_mves_mix':
            print('er_mves_mix')
            er_sortidx = np.argsort(er)
            er_idx = er_sortidx[-params.num_diversity*ack_batch_size:]
            best_pred = torch.cat([preds[:, er_idx], sorted_preds[:, -1].unsqueeze(-1)], dim=-1)
        elif params.measure == 'er_condense':
            print('er_condense')
            er_sortidx = np.argsort(er)
            er_idx = er_sortidx[-params.num_diversity*ack_batch_size:]
            best_pred = preds[:, er_idx]
            opt_weighting = torch.tensor((er[er_idx]-er[er_idx].min()))
        elif params.measure == 'er_pdts_mix':
            print('er_pdts_mix')
            er_sortidx = np.argsort(er)
            pdts_idx = bopt.get_pdts_idx(preds, params.num_diversity*ack_batch_size, density=True)
            print('pdts_idx:', pdts_idx)
            er_idx = er_sortidx[-params.num_diversity*ack_batch_size:]
            best_pred = torch.cat([preds[:, er_idx], preds[:, pdts_idx]], dim=-1)
        elif params.measure == 'cma_es':
            print('cma_es')
            indices = torch.LongTensor(list(skip_idx)).to(params.device)
            sortidx = torch.sort(Y[indices])[1]
            indices = indices[sortidx]
            assert indices.ndimension() == 1
            best_pred = preds[:, indices[-10:]]
            er_sortidx = np.argsort(er)
            er_idx = er_sortidx[-params.num_diversity*ack_batch_size:]
        elif params.measure == 'mves':
            print('mves')
            er_sortidx = np.argsort(er)
            er_idx = er_sortidx[-params.num_diversity*ack_batch_size:]
            best_pred = sorted_preds[:, -top_k:]
        else:
            assert False

        print('best_pred.shape\t' + str(best_pred.shape))
        #f.write('best_pred.shape\t' + str(best_pred.shape))

        print('best_pred:', best_pred.mean(0).mean(), best_pred.std(0).mean())

        condense_idx, best_hsic = acquire_batch_mves_sid(
                params,
                best_pred, 
                preds, 
                skip_idx, 
                params.mves_compute_batch_size, 
                ack_batch_size, 
                #true_labels=labels, 
                greedy_ordering=params.mves_greedy, 
                pred_weighting=params.pred_weighting, 
                normalize=True, 
                divide_by_std=params.divide_by_std, 
                opt_weighting=None,
                min_hsic_increase=params.min_hsic_increase,
                double=True,
                )

        print('er_idx', er_idx)
        print('condense_idx', condense_idx)
        print('intersection size', len(set(condense_idx).intersection(set(er_idx.tolist()))))

        if len(condense_idx) < ack_batch_size:
            for idx in er_idx[::-1]:
                idx = int(idx)
                if len(condense_idx) >= ack_batch_size:
                    break
                if idx in condense_idx and idx not in skip_idx:
                    continue
                condense_idx += [idx]
        assert len(condense_idx) == ack_batch_size
        assert len(set(condense_idx)) == ack_batch_size

        print('er_labels', labels[er_idx])
        print('mves_labels', labels[list(condense_idx)])

        print('best_hsic\t' + str(best_hsic))

        #f.write('best_hsic\t' + str(best_hsic) + "\n")
        #f.write('train_X.shape\t' + str(train_X_mves.shape) + "\n")

        return condense_idx

def pairwise_logging_pre_ack(
    params,
    preds,
    preds_vars,
    train_idx,
    ack_idx,
    unseen_idx,
    Y,
    do_hsic_rand=True,
    do_mi_rand=True,
    do_batch_hsic=True,
    hsic_custom_kernel=None,
):
    with torch.no_grad():
        hsic_rand = None
        mi_rand = None
        ack_batch_hsic = None
        rand_batch_hsic = None
        rand_idx = None
        unseen_idx_rand_set = unseen_idx
        unseen_idx_rand = list(unseen_idx_rand_set)

        if do_batch_hsic:
            rand_idx = np.random.choice(unseen_idx_rand, 300, replace=False).tolist()
            #ack_batch_hsic = max_batch_hsic(params, preds[:, list(ack_idx)], preds[:, rand_idx])
            ack_batch_hsic = hsic_with_batch(params, preds[:, list(ack_idx)], preds[:, rand_idx], hsic_custom_kernel=hsic_custom_kernel)

            #ack_batch_hsic = hsic_with_batch(params, preds[:, list(ack_idx)], preds[:, unseen_idx_rand])
            #rand_idx = torch.sort(ack_batch_hsic, descending=True)[1].cpu().numpy().tolist()[:300]
            #ack_batch_hsic = ack_batch_hsic[rand_idx]
            #rand_idx = np.array(unseen_idx_rand, dtype=np.int32)[rand_idx].tolist()

            rand_batch_idx = np.random.choice(list(unseen_idx_rand_set.difference(ack_idx)), len(ack_idx), replace=False).tolist()
            #rand_batch_hsic = max_batch_hsic(params, preds[:, rand_batch_idx], preds[:, rand_idx])
            rand_batch_hsic = hsic_with_batch(params, preds[:, rand_batch_idx], preds[:, rand_idx], hsic_custom_kernel=hsic_custom_kernel)

        else:
            rand_idx = np.random.choice(unseen_idx_rand, 300, replace=False).tolist()

        corr_rand = np.corrcoef(preds[:, rand_idx].transpose(0, 1).detach().cpu().numpy())
        if do_hsic_rand:
            hsic_rand = pairwise_hsic(params, preds[:, rand_idx])
        if do_mi_rand:
            mi_rand = pairwise_mi(params, preds[:, rand_idx])
        old_std = preds.std(dim=0)[rand_idx]

        old_nll = get_nll(
                preds[:, rand_idx], 
                preds_vars[:, rand_idx], 
                Y[rand_idx],
                params.output_noise_dist_fn
                )

        return rand_idx, old_std, old_nll, corr_rand, hsic_rand, mi_rand, ack_batch_hsic, rand_batch_hsic


def pairwise_logging_post_ack(
    params,
    preds,
    preds_vars,
    old_std,
    old_nll,
    corr_rand,
    Y,
    hsic_rand=None,
    mi_rand=None,
    ack_batch_hsic=None,
    rand_batch_hsic=None,
    stats_pred=None,
):

    with torch.no_grad():
        new_std = preds.std(dim=0)
        std_ratio_vector = old_std/new_std
        corr_rand = corr_rand[np.tril_indices(corr_rand.shape[0], k=-1)]
        std_ratio_matrix = compute_std_ratio_matrix(old_std, preds)

        new_nll = get_nll(
                preds, 
                preds_vars, 
                Y,
                params.output_noise_dist_fn
                )

        nll_diff = old_nll-new_nll

        corr_stats = [[], []]

        p1 = pearsonr(corr_rand, std_ratio_matrix)[0]
        kt = kendalltau(corr_rand, std_ratio_matrix)[0]
        print('corr pearson:', p1, ';', 'kt:', kt)
        corr_stats[0] += [float(p1)]
        corr_stats[1] += [float(kt)]

        if hsic_rand is not None:
            hsic_rand = hsic_rand[np.tril_indices(hsic_rand.shape[0], k=-1)]
            p1 = pearsonr(hsic_rand, std_ratio_matrix)[0]
            kt = kendalltau(hsic_rand, std_ratio_matrix)[0]
            print('hsic pearson:', p1, ';', 'kt:', kt)
            corr_stats[0] += [float(p1)]
            corr_stats[1] += [float(kt)]
        else:
            corr_stats[0] += [0]
            corr_stats[1] += [0]

        if mi_rand is not None:
            mi_rand = mi_rand[np.tril_indices(mi_rand.shape[0], k=-1)]
            p1 = pearsonr(mi_rand, std_ratio_matrix)[0]
            kt = kendalltau(mi_rand, std_ratio_matrix)[0]
            print('mi pearson:', p1, ';', 'kt:', kt)
            corr_stats[0] += [float(p1)]
            corr_stats[1] += [float(kt)]
        else:
            corr_stats[0] += [0]
            corr_stats[1] += [0]

        if ack_batch_hsic is not None:
            p1 = pearsonr(ack_batch_hsic, std_ratio_vector)[0]
            kt = kendalltau(ack_batch_hsic, std_ratio_vector)[0]
            print('ack_batch_hsic pearson:', p1, ';', 'kt:', kt)
            corr_stats[0] += [float(p1)]
            corr_stats[1] += [float(kt)]
        else:
            corr_stats[0] += [0]
            corr_stats[1] += [0]

        if rand_batch_hsic is not None:
            p1 = pearsonr(rand_batch_hsic, std_ratio_vector)[0]
            kt = kendalltau(rand_batch_hsic, std_ratio_vector)[0]
            print('rand_batch_hsic pearson:', p1, ';', 'kt:', kt)
            corr_stats[0] += [float(p1)]
            corr_stats[1] += [float(kt)]
        else:
            corr_stats[0] += [0]
            corr_stats[1] += [0]

        if ack_batch_hsic is not None:
            #pos_nll_index = (nll_diff >= 0).byte().cpu()
            #pos_nll_diff = torch.masked_select(nll_diff.cpu(), pos_nll_index)
            p1 = pearsonr(ack_batch_hsic, nll_diff)[0]
            kt = kendalltau(ack_batch_hsic, nll_diff)[0]
            #p1 = pearsonr(torch.masked_select(ack_batch_hsic.cpu(), pos_nll_index), pos_nll_diff)[0]
            #kt = kendalltau(torch.masked_select(ack_batch_hsic.cpu(), pos_nll_index), pos_nll_diff)[0]
            print('ack_batch_hsic pearson (nll):', p1, ';', 'kt:', kt)
            corr_stats[0] += [float(p1)]
            corr_stats[1] += [float(kt)]
        else:
            corr_stats[0] += [0]
            corr_stats[1] += [0]

        if rand_batch_hsic is not None:
            p1 = pearsonr(rand_batch_hsic, nll_diff)[0]
            kt = kendalltau(rand_batch_hsic, nll_diff)[0]
            print('rand_batch_hsic pearson (nll):', p1, ';', 'kt:', kt)
            corr_stats[0] += [float(p1)]
            corr_stats[1] += [float(kt)]
        else:
            corr_stats[0] += [0]
            corr_stats[1] += [0]

        if ack_batch_hsic is not None:
            p1 = ack_batch_hsic.mean().item()
            kt = ack_batch_hsic.std().item()
            print('ack_batch_hsic mean:', p1, ';', 'stddev:', kt)
            corr_stats[0] += [float(p1)]
            corr_stats[1] += [float(kt)]

            print('ack_batch_hsic max:', ack_batch_hsic.max().item(), ';', 'min:', ack_batch_hsic.min().item())
        else:
            corr_stats[0] += [0]
            corr_stats[1] += [0]

        if rand_batch_hsic is not None:
            p1 = rand_batch_hsic.mean().item()
            kt = rand_batch_hsic.std().item()
            print('rand_batch_hsic mean:', p1, ';', 'stddev:', kt)
            corr_stats[0] += [float(p1)]
            corr_stats[1] += [float(kt)]
        else:
            corr_stats[0] += [0]
            corr_stats[1] += [0]

        if stats_pred is not None:
            assert len(stats_pred) == 3
            if stats_pred[0] is None:
                corr_stats[0] += [0]
                corr_stats[1] += [0]
            else:
                p1 = pearsonr(old_std-new_std, stats_pred[0])[0]
                kt = kendalltau(old_std-new_std, stats_pred[0])[0]
                print('predict_std pearson:', p1, ';', 'kt:', kt)
                corr_stats[0] += [float(p1)]
                corr_stats[1] += [float(kt)]

            if stats_pred[1] is None:
                corr_stats[0] += [0]
                corr_stats[1] += [0]
            else:
                assert False, "Not implemented"

            if stats_pred[2] is None:
                corr_stats[0] += [0]
                corr_stats[1] += [0]
            else:
                p1 = pearsonr(nll_diff, stats_pred[2])[0]
                kt = kendalltau(nll_diff, stats_pred[2])[0]
                print('predict_nll pearson:', p1, ';', 'kt:', kt)
                corr_stats[0] += [float(p1)]
                corr_stats[1] += [float(kt)]


        return corr_stats

def compute_ack_stat_change_val_nll_ensemble(
    params,
    X,
    unscaled_Y,
    preds,
    ack_idx,
    model,
    lr,
    l2,
    seen_idx,
    normalize_fn=None,
    seen_batch_size=0,
):
    from deep_ensemble_sid import NNEnsemble
    assert len(set(seen_idx).intersection(ack_idx)) == 0
    seen_idx = np.array(list(seen_idx))

    N = len(seen_idx)
    val_frac = params.empirical_stat_val_fraction
    train_idx, val_idx, _, _ = utils.train_val_test_split(
            N, 
            [1-val_frac, val_frac],
            )
    train_idx = seen_idx[train_idx].tolist()
    idx_to_monitor = seen_idx[val_idx].tolist()

    train_X = X[train_idx]
    train_Y = unscaled_Y[train_idx]
    monitor_X = X[idx_to_monitor]

    if params.sigmoid_coeff > 0:
        Y = utils.sigmoid_standardization(
                unscaled_Y,
                train_Y.mean(),
                train_Y.std(),
                exp=torch.exp)
    else:
        Y = utils.normal_standardization(
                unscaled_Y,
                train_Y.mean(),
                train_Y.std(),
                exp=torch.exp)

    train_Y = Y[train_idx]
    monitor_Y = Y[idx_to_monitor]


    seen_batch_size = min(seen_batch_size, train_X.shape[0])

    if params.empirical_stat == "val_classify":
        monitor_sorted_idx = torch.sort(monitor_Y)[1]

    if normalize_fn is not None:
        mean = train_Y.mean()
        std = train_Y.std()
        train_Y = normalize_fn(train_Y, mean, std, exp=torch.exp)
        monitor_Y = normalize_fn(monitor_Y, mean, std, exp=torch.exp)

    with torch.no_grad():
        monitor_means, monitor_variances = model(monitor_X)

        if params.empirical_stat == "val_nll":
            nll_mixture, nll_single_gaussian = NNEnsemble.report_metric(
                    monitor_Y,
                    monitor_means,
                    monitor_variances,
                    custom_std=train_Y.std() if params.report_metric_train_std else None,
                    return_mse=False)
            if params.single_gaussian_test_nll:
                stat = nll_single_gaussian
            else:
                stat = nll_mixture
        elif params.empirical_stat == "val_classify":
            kt_labels = torch.arange(monitor_X.shape[0])[monitor_sorted_idx]
            mean_of_means = monitor_means.mean(dim=0)
            classify_kt_corr = kendalltau(mean_of_means[monitor_sorted_idx], kt_labels)[0]
            stat = -torch.tensor(classify_kt_corr, device=params.device)
        else:
            assert False, "params.empirical_stat %s not implemented" % (params.empirical_stat,)

    #print('X.shape', X.shape, ack_idx)
    if seen_batch_size == 0:
        bX = X[ack_idx + train_idx]
        rand_idx = train_idx
    else:
        rand_idx = utils.randint(len(train_idx), seen_batch_size)
        rand_idx = np.array(train_idx)[rand_idx].tolist()
        bX = X[ack_idx + rand_idx]

    if len(bX.shape) == 1:
        bX = bX.unsqueeze(dim=0)

    bY = preds[:, ack_idx].mean(dim=0)
    bY = torch.cat([bY, Y[rand_idx]], dim=0)

    optim = torch.optim.Adam(list(model.parameters()), lr=lr, weight_decay=l2)
    for i in range(1):
        means, variances = model(bX)
        nll = NNEnsemble.compute_negative_log_likelihood(
            bY,
            means,
            variances,
            return_mse=False,
        )
        optim.zero_grad()
        nll.backward()
        optim.step()

    with torch.no_grad():
        monitor_means2, monitor_variances2 = model(monitor_X)
        if params.empirical_stat == "val_nll":
            nll_mixture, nll_single_gaussian = NNEnsemble.report_metric(
                    monitor_Y,
                    monitor_means2,
                    monitor_variances2,
                    custom_std=train_Y.std() if params.report_metric_train_std else None,
                    return_mse=False)
            if params.single_gaussian_test_nll:
                stat2 = nll_single_gaussian
            else:
                stat2 = nll_mixture
        elif params.empirical_stat == "val_classify":
            kt_labels = torch.arange(monitor_X.shape[0])[monitor_sorted_idx]
            mean_of_means = monitor_means2.mean(dim=0)
            classify_kt_corr = kendalltau(mean_of_means[monitor_sorted_idx], kt_labels)[0]
            stat2 = -torch.tensor(classify_kt_corr, device=params.device)
        else:
            assert False, "params.empirical_stat %s not implemented" % (params.empirical_stat,)

    return stat2-stat, stat



def compute_ack_stat_change(
    params,
    X,
    unscaled_Y,
    preds,
    ack_idx,
    model,
    lr,
    l2,
    seen_idx,
    idx_to_monitor,
    stat_fn, # (num_samples, indices) -> stddev/mmd to uniform/etc
    seen_batch_size=0,
    stat_fn_info=None,
):
    #assert len(set(seen_idx).intersection(idx_to_monitor)) == 0
    assert len(set(seen_idx).intersection(ack_idx)) == 0

    seen_idx = list(seen_idx)
    optim = torch.optim.Adam(list(model.parameters()), lr=lr, weight_decay=l2)
    stat = stat_fn(preds[:, idx_to_monitor], stat_fn_info)

    #print('X.shape', X.shape, ack_idx)
    if seen_batch_size == 0:
        bX = X[ack_idx + seen_idx]
        rand_idx = seen_idx
    else:
        rand_idx = utils.randint(len(seen_idx), seen_batch_size)
        rand_idx = np.array(seen_idx)[rand_idx].tolist()
        bX = X[ack_idx + rand_idx]

    if len(bX.shape) == 1:
        bX = bX.unsqueeze(dim=0)

    bY = unscaled_Y[rand_idx] 

    if params.sigmoid_coeff > 0:
        bY = utils.sigmoid_standardization(
                bY,
                bY.mean(),
                bY.std(),
                exp=torch.exp)
    else:
        bY = utils.normal_standardization(
                bY,
                bY.mean(),
                bY.std(),
                exp=torch.exp)
    bY = torch.cat([preds[:, ack_idx].mean(dim=0), bY], dim=0)

    for i in range(2):
        means, variances = model(bX)
        nll = model.compute_negative_log_likelihood(
            bY,
            means,
            variances,
            return_mse=False,
        )
        optim.zero_grad()
        nll.backward()
        optim.step()

    with torch.no_grad():
        means2, variances2 = model(X[idx_to_monitor])
        stat2 = stat_fn(means2, stat_fn_info)

    optim.zero_grad()

    return stat2-stat, stat



def get_empirical_condensation_ack(
    params,
    X,
    Y,
    preds,
    model,
    lr,
    l2,
    idx_to_condense,
    seen_idx,
    ack_batch_size,
    idx_to_monitor=None,
    er_values=None,
    stat_fn = lambda preds, info : preds.std(dim=0),
    stat_fn_info = None,
    predict_info_model=None,
    seen_batch_size=0,
    val_nll_metric=False,
):
    assert len(set(idx_to_condense).intersection(seen_idx)) == 0
    cur_ack_batch = []

    for ack_batch_iter in range(ack_batch_size):
        #print('ack_batch_iter:', ack_batch_iter)
        stat_change_list = torch.ones(X.shape[0])*1000
        #for idx in range(X.shape[0]):
        new_idx_to_condense = []
        for idx in idx_to_condense:
            if idx in seen_idx or idx in cur_ack_batch:
                continue
            new_idx_to_condense += [idx]
        max_stat_change = 0
        new_idx_to_condense = set(new_idx_to_condense)
        idx_stats = []
        for idx in new_idx_to_condense:
            model_copy = copy.deepcopy(model)
            if val_nll_metric:
                stat_change, old_stat = compute_ack_stat_change_val_nll_ensemble(
                        params,
                        X,
                        Y,
                        preds,
                        cur_ack_batch + [idx],
                        model_copy,
                        lr,
                        l2,
                        seen_idx,
                        seen_batch_size=seen_batch_size,
                        normalize_fn=utils.sigmoid_standardization if params.sigmoid_coeff > 0 else utils.normal_standardization,
                        )
            else:
                stat_change, old_stat = compute_ack_stat_change(
                        params,
                        X,
                        Y,
                        preds,
                        cur_ack_batch + [idx],
                        model_copy,
                        lr,
                        l2,
                        seen_idx,
                        list(new_idx_to_condense.difference({idx})) if idx_to_monitor is None else list(set(idx_to_monitor).difference({idx})),
                        stat_fn,
                        stat_fn_info=stat_fn_info,
                        seen_batch_size=seen_batch_size,
                        )
            stat_change_list[idx] = torch.mean(stat_change).item()
            idx_stats += [stat_change_list[idx]]

        #er_std_factor = float(np.std(idx_stats))/er_values.std()
        #for idx in new_idx_to_condense:
        #    stat_change_list[idx] -= (er_values[idx] * er_std_factor).item() 
        stat_change_sort, stat_change_sort_idx = torch.sort(stat_change_list)
        print('stat_change_sort:', stat_change_sort)
        cur_ack_batch += [stat_change_sort_idx[0].item()]

    return cur_ack_batch


def get_empirical_condensation_ack2(
    params,
    X,
    Y,
    preds,
    model,
    lr,
    l2,
    idx_to_condense,
    seen_idx,
    ack_batch_size,
    idx_to_monitor=None,
    er_values=None,
    stat_fn = lambda preds, info : preds.std(dim=0),
    stat_fn_info = None,
    predict_info_model=None,
    seen_batch_size=0,
    val_nll_metric=False,
):
    assert len(set(idx_to_condense).intersection(seen_idx)) == 0

    model_copy = copy.deepcopy(model)
    if val_nll_metric:
        stat_change, old_stat = compute_ack_stat_change_val_nll_ensemble(
                params,
                X,
                Y,
                preds,
                idx_to_condense,
                model_copy,
                lr,
                l2,
                seen_idx,
                seen_batch_size=seen_batch_size,
                normalize_fn=utils.sigmoid_standardization if params.sigmoid_coeff > 0 else utils.normal_standardization,
                )
    else:
        stat_change, old_stat = compute_ack_stat_change(
                params,
                X,
                Y,
                preds,
                idx_to_condense,
                model_copy,
                lr,
                l2,
                seen_idx,
                idx_to_condense,
                stat_fn,
                stat_fn_info=stat_fn_info,
                seen_batch_size=seen_batch_size,
                )

        #er_std_factor = float(np.std(idx_stats))/er_values.std()
        #for idx in new_idx_to_condense:
        #    stat_change_list[idx] -= (er_values[idx] * er_std_factor).item() 
    stat_change_sort, stat_change_sort_idx = torch.sort(stat_change)
    print('stat_change_sort[:10]:', stat_change_sort[:10])

    cur_ack_batch = [idx_to_condense[stat_change_sort_idx[i].item()] for i in range(ack_batch_size)]
    return cur_ack_batch


def get_empirical_condensation_ack3(
    params,
    X,
    Y,
    pred_means,
    pred_vars,
    orig_model,
    lr,
    l2,
    idx_to_condense,
    seen_idx,
    ack_batch_size,
    cur_rmse,
    idx_to_monitor=None,
    er_values=None,
    stat_fn = lambda pred_means, info : pred_means.std(dim=0),
    stat_fn_info = None,
    predict_info_model=None,
    seen_batch_size=0,
    val_nll_metric=False,
):
    assert len(set(idx_to_condense).intersection(seen_idx)) == 0

    seen_idx = list(seen_idx)
    stat = stat_fn(pred_means[:, idx_to_monitor], stat_fn_info)

    bX = X[idx_to_condense + seen_idx]

    if len(bX.shape) == 1:
        bX = bX.unsqueeze(dim=0)

    bY = Y[seen_idx] 
    if params.sigmoid_coeff > 0:
        bY = utils.sigmoid_standardization(
                bY,
                bY.mean(),
                bY.std(),
                exp=torch.exp)
    else:
        bY = utils.normal_standardization(
                bY,
                bY.mean(),
                bY.std(),
                exp=torch.exp)

    corr = []
    num_to_monitor = len(idx_to_condense)
    pred_matrix = []

    with torch.no_grad():
        old_std = torch.sqrt(pred_vars).mean().item()
        cur_rmse = max(cur_rmse-old_std, 0.)
        old_idx_to_condense_std = pred_means[:, idx_to_condense].std(dim=0)
        old_idx_to_condense_means = pred_means[:, idx_to_condense].mean(dim=0)
        old_rmse = torch.sqrt(((pred_means[:, seen_idx].mean(dim=0)-bY)**2).mean())

    for ack_point_iter in range(num_to_monitor):
        ack_point = idx_to_condense[ack_point_iter]
        num_iter = 4
        model = copy.deepcopy(orig_model)
        optim = torch.optim.Adam(list(model.parameters()), lr=lr, weight_decay=l2)

        bX2 = X[[ack_point] + seen_idx]
        bY2 = torch.cat([pred_means[:, ack_point].mean(dim=0).unsqueeze(0), bY], dim=0)
        temp = []
        for i in range(num_iter):
            means, variances = model(bX2)
            nll = model.compute_negative_log_likelihood(
                bY2,
                means,
                variances,
                return_mse=False,
            )
            optim.zero_grad()
            nll.backward()
            optim.step()

            with torch.no_grad():
                means, _ = model(X)
                temp  += [torch.max(means, dim=1)[0].std()]

        with torch.no_grad():
            #means, variances = model(bX)
            #corr += [old_idx_to_condense_std-means[:, :num_to_monitor]:.std(dim=0)]
            #corr += [-old_idx_to_condense_std.mean()+means[:, :num_to_monitor].std(dim=0).mean()]
            #new_rmse = torch.sqrt(((means[:, num_to_monitor:].mean(dim=0)-bY)**2).mean())
            #corr += [torch.abs(old_rmse-new_rmse)]
            #pred_matrix += [means]
            #means, _ = model(X)
            #max_vals_std = torch.max(means, dim=1)[0].std()
            #corr += [max_vals_std]
            corr += [torch.mean(torch.tensor(temp, device=params.device))]
    ucb_measure = old_idx_to_condense_means + params.ucb * old_idx_to_condense_std

    #pred_matrix = torch.cat(pred_matrix, dim=0)
    #pred_corr_matrix = ops.corrcoef(pred_matrix.transpose(0, 1))

    #pred_corr = pred_corr_matrix.mean(dim=1)

    corr = torch.stack(corr, dim=0)
    assert corr.shape[0] == num_to_monitor
    #corr_matrix = ops.corrcoef(corr.transpose(0, 1))

    """
    if params.ekb_use_median:
        #corr = corr_matrix.median(dim=1)[0]
        corr = corr.median(dim=1)[0]
        assert corr.shape[0] == num_to_monitor
    else:
        #corr = corr_matrix.mean(dim=1)
        corr = corr.mean(dim=1)
        assert corr.shape[0] == num_to_monitor
    """

    if params.empirical_diversity_only:
        stat_to_sort = ucb_measure
    else:
        corr = (corr-corr.mean())/(corr.std()+0.0001)
        stat_to_sort = -params.ucb_ekb_weighting*cur_rmse*corr + ucb_measure
        #stat_to_sort = corr + ucb_measure
    stat_change_sort, stat_change_sort_idx = torch.sort(stat_to_sort, descending=True)
    print('stat_change_sort[:10]:', stat_change_sort[:10])
    stat_change_sort_idx = [k.item() for k in stat_change_sort_idx]

    #cur_ack_batch = [idx_to_condense[stat_change_sort_idx[i].item()] for i in range(ack_batch_size)]
    cur_ack_batch = []
    cur_sort_idx = stat_change_sort_idx
    for i in range(len(stat_change_sort_idx)):
        if i == 0:
            cur_ack_batch += [stat_change_sort_idx[0]]
        else:
            #m = corr_matrix[stat_change_sort_idx[i], cur_ack_batch].max()
            #if m >= 0.9:
            #    continue
            cur_ack_batch += [cur_sort_idx[i]]
            #cur_stat_to_sort = [1000]*(i+1) + [stat_to_sort[idx]-0.1*corr_matrix[idx, cur_ack_batch].mean() for idx in stat_change_sort_idx[i+1:]]
            #_, cur_sort_idx = torch.sort(stat_to_sort.new_tensor(cur_stat_to_sort), descending=True)
        if len(cur_ack_batch) == ack_batch_size:
            break
    cur_ack_batch = [idx_to_condense[k] for k in cur_ack_batch]
    return cur_ack_batch


def compute_pearson(preds, info):
    return ops.corrcoef(preds[info['rand_idx']].transpose(0, 1)).view(-1)


def compute_maxdist_entropy(preds, info, k=5, device='cuda'):
    #h = info.get_h(preds.max(dim=1)[0].unsqueeze(dim=-1).cpu().numpy(), k)
    #return torch.tensor(h, device=device)
    return preds.max(dim=1)[0].std()


def get_best_ucb_beta(
    pre_ack_preds,
    Y,
    ucb_beta_range,
):
    best_kt = -1
    best_ucb = None

    Y = Y.cpu().numpy()
    er = pre_ack_preds.mean(dim=0).cpu().numpy()
    std = pre_ack_preds.std(dim=0).cpu().numpy()
    for ucb_beta in ucb_beta_range:
        ucb = er + (ucb_beta * std)
        kt = kendalltau(ucb, Y)[0]

        if kt >= best_kt-0.001:
            best_kt = kt
            best_ucb = float(ucb_beta)
    return best_ucb, best_kt


def predict_ood_rand_pairs(
    init_size,
    ack_size,
    num_acks_done,
    num_pairs_needed,
    idx_at_each_iter,
    only_next_ack=True
):
    idx_at_each_iter = [idx for idx_list in idx_at_each_iter for idx in idx_list]

    if only_next_ack:
        num_total_pairs_per_ack = [(init_size+ack_size*i) * ack_size for i in range(num_acks_done)]
    else:
        num_total_pairs_per_ack = [(init_size+ack_size*i) * (ack_size * (num_acks_done-i)) for i in range(num_acks_done)]

    cumsum_total_pairs = [sum(num_total_pairs_per_ack[:i]) for i in range(len(num_total_pairs_per_ack)+1)]
    total_pairs = sum(num_total_pairs_per_ack)
    num_pairs_needed = min(total_pairs, num_pairs_needed)

    pairs = [ [[], []] for i in range(num_acks_done)]
    pair_idx_list = np.random.choice(total_pairs, size=num_pairs_needed, replace=False)
    for i in range(num_pairs_needed):
        pair_idx = pair_idx_list[i]
        ack_iter = utils.cumsub(pair_idx, num_total_pairs_per_ack)
        pair_idx -= cumsum_total_pairs[ack_iter]
        assert pair_idx >= 0, "%d, %d, %d" % (ack_iter, pair_idx, num_acks_done)
        assert pair_idx < num_total_pairs_per_ack[ack_iter], "%d, %d" % (ack_iter, pair_idx)

        if only_next_ack:
            num_not_seen = ack_size
        else:
            num_not_seen = ack_size * (num_acks_done-ack_iter)

        pairs[ack_iter][0] += [pair_idx//num_not_seen]
        pairs[ack_iter][1] += [pair_idx%num_not_seen]
        pairs[ack_iter][1][-1] = init_size + (ack_size*ack_iter) + pairs[ack_iter][1][-1]

    return pairs, num_pairs_needed


def train_ood_pred(
    params,
    ood_pred_model,
    model,
    all_pred_means,
    all_pred_variances,
    X,
    unscaled_Y,
    idx_at_each_iter,
):
    num_acks_done = len(all_pred_means)
    idx_flatten = [idx for idx_list in idx_at_each_iter for idx in idx_list]
    assert num_acks_done+1 == len(idx_at_each_iter), "%d == %d" % (len(all_preds), len(idx_at_each_iter))
    assert num_acks_done == len(all_pred_variances)
    optim = torch.optim.Adam(list(ood_pred_model.parameters()), lr=params.ood_pred_lr)

    nll_list = []
    with torch.no_grad():
        for ack_iter in range(num_acks_done):
            mean_preds = all_pred_means[ack_iter]
            variance_preds = all_pred_variances[ack_iter]

            train_Y = unscaled_Y[idx_flatten[:len(idx_at_each_iter[0])+params.ack_batch_size*ack_iter]]
            if params.sigmoid_coeff > 0:
                Y = utils.sigmoid_standardization(
                        unscaled_Y,
                        train_Y.mean(),
                        train_Y.std(),
                        exp=torch.exp)
            else:
                Y = utils.normal_standardization(
                        unscaled_Y,
                        train_Y.mean(),
                        train_Y.std(),
                        exp=torch.exp)

            nll = model.get_per_point_nll(
                    Y[idx_flatten[:len(idx_at_each_iter[0])+params.ack_batch_size*(ack_iter+1)]],
                    mean_preds,
                    variance_preds,
                    )
            nll_list += [nll]

    for epoch_iter in range(params.ood_pred_epoch_iter):
        pairs, _ = predict_ood_rand_pairs(
                params.init_train_examples, 
                params.ack_batch_size,
                num_acks_done,
                params.ood_pred_batch_size,
                idx_at_each_iter,
                )

        assert len(pairs) == num_acks_done
        loss = 0
        for ack_iter in range(num_acks_done):

            # TODO -- running on all idx. Could be made more efficient by running only on those selected by rand_pairs
            # Otoh -- maybe rand_pairs should selected every idx at least once...
            emb = ood_pred_model(X[idx_flatten])
            pairs_emb = [emb[pairs[ack_iter][0]], emb[pairs[ack_iter][1]]]

            nll_preds = torch.sum((pairs_emb[1]-pairs_emb[0])**2, dim=1)
            assert nll_preds.shape[0] <= params.ood_pred_batch_size

            nll = nll_list[ack_iter][pairs[ack_iter][1]]

            loss += torch.mean((nll_preds-nll)**2)

        loss /= num_acks_done

        optim.zero_grad()
        loss.backward()
        optim.step()


def predict_ood(
    ood_pred_model,
    seen_idx,
    ack_idx,
    X,
):
    seen_emb = ood_pred_model(X[seen_idx]) # (n, emb_size)
    ack_emb = ood_pred_model(X[ack_idx]) # (m, emb_size)

    dist_matrix = utils.sqdist(seen_emb.unsqueeze(1), ack_emb.unsqueeze(1))[:, :, 0] # (n, m)
    nll_pred = torch.mean(dist_matrix, dim=0)

    return nll_pred


def metalearn(
    params,
    model,
    X,
    Y,
    seen_idx,
    rollout_length=5,
):
    assert False, "Not implemented"
    seen_idx = list(seen_idx)

    model = copy.deepcopy(model)
    optim = torch.optim.Adam(list(model.parameters()), lr=params.re_train_lr, weight_decay=params.re_train_l2)

    for ack_iter in range(rollout_length):
        pass
