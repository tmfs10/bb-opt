import os
import gc
import sys
import numpy as np
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


def ei_diversity_selection_hsic(
    params,
    preds, #(num_samples, num_candidate_points)
    skip_idx_ei,
    num_ei=100,
    device = 'cuda',
    ucb=False,
):
    ack_batch_size = params.ack_batch_size
    kernel_fn = getattr(hsic, params.mves_kernel_fn)

    ei = preds.mean(dim=0).view(-1).cpu().numpy()
    ei_sortidx = np.argsort(ei)
    temp = []
    for idx in ei_sortidx[::-1]:
        if idx not in skip_idx_ei:
            temp += [idx]
        if len(temp) == num_ei:
            break
    ei_sortidx = np.array(temp)
    ei = ei[ei_sortidx]
    ei -= ei.min().item()
    preds = preds[:, ei_sortidx]

    chosen_org_idx = [ei_sortidx[-1]]

    chosen_idx = [num_ei-1]
    batch_dist_matrix = hsic.sqdist(preds[:, -1].unsqueeze(-1)) # (n, n, 1)
    self_kernel_matrix = kernel_fn(batch_dist_matrix).detach().repeat([1, 1, 2])
    normalizer = torch.log(hsic.total_hsic(self_kernel_matrix))

    all_dist_matrix = hsic.sqdist(preds) # (n, n, m)
    all_kernel_matrix = kernel_fn(all_dist_matrix) # (n, n, m)
    self_kernel_matrix = all_kernel_matrix.permute([2, 0, 1]).unsqueeze(-1).repeat([1, 1, 1, 2]) # (m, n, n, 2)
    hsic_var = hsic.total_hsic_parallel(self_kernel_matrix)
    hsic_logvar = torch.log(hsic_var).view(-1)

    while len(chosen_idx) < ack_batch_size:
        if ucb:
            dist_matrix = all_dist_matrix + batch_dist_matrix
            kernel_matrix = kernel_fn(dist_matrix).detach().permute([2, 0, 1]).unsqueeze(-1).repeat([1, 1, 1, 2]) # (m, n, n, 2)
            hsic_covar = hsic.total_hsic_parallel(kernel_matrix).detach()
        else:
            a = len(chosen_idx)
            m = all_kernel_matrix.shape[-1]
            kernel_matrix = all_kernel_matrix[:, :, chosen_idx].unsqueeze(-1).repeat([1, 1, 1, m]).permute([3, 0, 1, 2]) # (m, n, n, a)
            kernel_matrix = torch.cat([kernel_matrix, all_kernel_matrix.permute([2, 0 ,1]).unsqueeze(-1)], dim=-1)
            hsic_covar = hsic.total_hsic_parallel(kernel_matrix).detach()

        normalizer = torch.exp((hsic_logvar + hsic_logvar[chosen_idx].sum())/(len(chosen_idx)+1))
        hsic_corr = hsic_covar/normalizer

        hsic_ucb = ei/hsic_corr
        hsic_ucb_sort_idx = hsic_ucb.detach().cpu().numpy().argsort()

        for idx in hsic_ucb_sort_idx[::-1]:
            if idx not in chosen_idx:
                break

        choice = ei_sortidx[idx]
        chosen_idx += [idx]
        chosen_org_idx += [choice]
        batch_dist_matrix += all_dist_matrix[:, :, idx:idx+1]

    return list(chosen_org_idx)


def ei_diversity_selection_detk(
    params,
    preds, #(num_samples, num_candidate_points)
    skip_idx_ei,
    num_ei=100,
    device = 'cuda',
    do_correlation=True,
    add_I=False,
    do_kernel=True,
):
    ack_batch_size = params.ack_batch_size

    ei = preds.mean(dim=0).view(-1).cpu().numpy()
    ei_sortidx = np.argsort(ei)
    temp = []
    for idx in ei_sortidx[::-1]:
        if idx not in skip_idx_ei:
            temp += [idx]
        if len(temp) == num_ei:
            break
    ei_sortidx = np.array(temp)
    ei = ei[ei_sortidx]
    ei -= ei.min().item()

    if do_kernel:
        preds = preds[:, ei_sortidx]
        kernel_fn = getattr(hsic, params.mves_kernel_fn)
        dist_matrix = hsic.sqdist(preds.transpose(0, 1).unsqueeze(1)) # (m, m, 1)
        covar_matrix = kernel_fn(dist_matrix)[:, :, 0] # (m, m)
    else:
        preds = preds[:, ei_sortidx].cpu().numpy()
        covar_matrix = np.cov(preds)
        covar_matrix = torch.FloatTensor(covar_matrix).to(device) # (num_candidate_points, num_candidate_points)

    if add_I:
        covar_matrix += torch.eye(covar_matrix.shape[0])
    if do_correlation:
        variances = torch.diag(covar_matrix)
        normalizer = torch.sqrt(variances.unsqueeze(0)*variances.unsqueeze(1))
        covar_matrix /= normalizer

    chosen_org_idx = [ei_sortidx[-1]]
    chosen_idx = [num_ei-1]
    while len(chosen_idx) < ack_batch_size:
        K_values = []
        num_already_chosen = len(chosen_idx)

        for idx in range(num_ei-1):
            if idx in chosen_idx:
                K_values += [1e-9]
                continue

            cur_chosen = torch.LongTensor(chosen_idx+[idx])
            logdet = torch.logdet(covar_matrix[cur_chosen, :][:, cur_chosen]).item()
            K_values += [logdet]

        K_values = np.array(K_values)

        K_ucb = ei[:-1]/K_values
        K_ucb_sort_idx = K_ucb.argsort()

        choice = ei_sortidx[K_ucb_sort_idx[-1]]
        chosen_idx += [K_ucb_sort_idx[-1]]
        chosen_org_idx += [choice]

    return chosen_idx

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

            batch_kernel_matrix = getattr(hsic, params.mves_kernel_fn)(dist_matrix) \
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
    opt_kernel_matrix = getattr(hsic, params.mves_kernel_fn)(opt_dist_matrix).detach()  # shape (n=num_samples, n, 1)

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

            batch_kernel_matrix = getattr(hsic, params.mves_kernel_fn)(dist_matrix).detach().permute([2, 0, 1]).unsqueeze(-1) # (m, n, n, 1)
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
    kernel_fn = getattr(hsic, 'two_vec_' + params.mves_kernel_fn)

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
        if hsic_diversity_lambda > 1e-9:
            kernels = hsic.dimwise_mixrq_kernels(preds)
            total_hsic = hsic.total_hsic(kernels)

            if normalize_hsic:
                normalizer = hsic.total_hsic_parallel(kernels.permute(2, 0, 1).unsqueeze(-1).repeat([1, 1, 1, 2])).view(-1)
                normalizer = torch.exp(normalizer.log().sum()*0.5)
                total_hsic /= normalizer

            loss += hsic_diversity_lambda*total_hsic
            postfix['hsic_loss'] = total_hsic.item()
            postfix['hsic_loss_real'] = (hsic_diversity_lambda*total_hsic).item()
        else:
            kernels = hsic.dimwise_mixrq_kernels(preds)
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
    jupyter=False,
):
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
        preds, _ = model_ensemble(input_tensor2, all_pairs=False) # (num_samples,)
        assert preds.ndimension() == 1
        assert preds.shape[0] == num_points_to_optimize

        loss = -torch.mean(preds)

        optim.zero_grad()
        loss.backward()
        optim.step()
        progress.set_description("normal_loss: %2.6f" % (loss.item()))

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
    assert opt_values_kernel_matrix.shape[2] == 1

    new_batch_matrix = kernel_fn(X, X, do_mean=do_mean)  # return is of shape (n=num_samples, n, 1)
    kernels = torch.cat([new_batch_matrix, opt_values_kernel_matrix], dim=-1)

    return hsic.total_hsic(kernels)


def acquire_batch_via_grad_hsic(
    params,
    model_ensemble: Callable[[torch.tensor], torch.tensor],
    input_shape: List[int],
    opt_values: torch.tensor, # (num_samples, preds)
    ack_batch_size,
    do_mean=False,
    seed: torch.tensor = None,
    normalize_hsic=True,
    one_hot=True,
    input_transform=lambda x : x,
    jupyter: bool = False,
) -> torch.tensor:

    print('mves: ack_batch_size', ack_batch_size)
    if seed is None:
        input_tensor = torch.randn([ack_batch_size] + input_shape, device=params.device, requires_grad=True)
    else:
        assert seed.shape[0] == ack_batch_size
        input_tensor = torch.tensor(seed, device=device, requires_grad=True)
    optim = torch.optim.Adam([input_tensor], lr=params.hsic_opt_lr)

    kernel_fn = getattr(hsic, "two_vec_" + params.hsic_kernel_fn)
    opt_kernel_matrix = kernel_fn(opt_values, opt_values, do_mean=do_mean)  # shape (n=num_samples, n, 1)
    opt_normalizer = torch.log(hsic.total_hsic(opt_kernel_matrix.repeat([1, 1, 2]))).view(-1)
    opt_normalizer_exp = torch.exp(0.5 * opt_normalizer).detach().item()

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
        assert preds.ndimension() == 2
        assert opt_kernel_matrix.shape[0] == preds.shape[0], str(opt_kernel_matrix.shape) + "[0] == " + str(preds.shape) + "[0]"

        total_hsic = two_dist_hsic(preds, opt_kernel_matrix, kernel_fn, do_mean)

        if normalize_hsic:
            total_hsic = total_hsic.log()
            self_kernel_matrix = hsic.two_vec_mixrq_kernels(preds)
            self_hsic_log = hsic.total_hsic(self_kernel_matrix.repeat([1, 1, 2])).log()
            total_hsic = total_hsic - 0.5*(self_hsic_log+opt_normalizer)
            total_hsic = total_hsic.exp()

        loss = -total_hsic

        progress.set_description("hsic_loss: %1.5f" % (loss.item()))

        optim.zero_grad()
        loss.backward()
        optim.step()

    return input_tensor.detach()


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


def get_pred_stats(preds, std, Y, unscaled_Y, output_dist_fn, test_idx, single_gaussian=True):
    assert preds.shape[1] == Y.shape[0]
    assert preds.shape == std.shape, "%s == %s" % (str(preds.shape), str(std.shape))

    preds = preds[:, test_idx]
    std = std[:, test_idx]
    Y = Y[test_idx]
    unscaled_Y = unscaled_Y[test_idx]

    log_prob_list = []
    mse_list = []
    kt_corr_list = []
    std_list = []
    mse_std_corr = []

    for frac in [1., 0.1]:
        labels_sort_idx = torch.sort(unscaled_Y, descending=True)[1].cpu().numpy()
        n = int(labels_sort_idx.shape[0] * frac)
        m = preds.shape[0]
        labels_sort_idx = labels_sort_idx[:n]
        labels2 = Y[labels_sort_idx]

        if single_gaussian:
            output_dist = output_dist_fn(
                    preds[:, labels_sort_idx].mean(0),
                    std[:, labels_sort_idx].mean(0))
            log_prob_list += [torch.mean(-output_dist.log_prob(labels2)).item()]
        else:
            output_dist = output_dist_fn(
                    preds[:, labels_sort_idx].view(-1),
                    std[:, labels_sort_idx].view(-1))
            log_prob_list += [torch.mean(-output_dist.log_prob(labels2.repeat([m]))).item()]

        pred_means = preds[:, labels_sort_idx].mean(0)
        mse = (pred_means-labels2)**2
        mse_list += [torch.sqrt(torch.mean(mse)).item()]
        kt_corr_list += [kendalltau(pred_means, unscaled_Y[labels_sort_idx])[0]]
        pred_std = preds[:, labels_sort_idx].std(0)
        std_list += [pred_std.mean().item()]
        mse_std_corr += [float(pearsonr(torch.sqrt(mse).cpu().numpy(), pred_std.cpu().numpy())[0])]

    max_idx = labels_sort_idx[0]
    log_prob_list += [torch.mean(-output_dist_fn(preds[:, max_idx], std[:, max_idx]).log_prob(Y[max_idx])).item()]
    mse_list += [torch.abs(preds[:, max_idx].mean(0)-Y[max_idx]).item()]
    kt_corr_list += [0]
    mse_std_corr += [0]
    std_list += [preds[:, max_idx].std(0).item()]

    return log_prob_list, mse_list, kt_corr_list, std_list, mse_std_corr

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
