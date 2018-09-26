import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as get_fingerprint
import torch
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
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Tuple, Optional, Callable, Dict, Set, List, Any, Iterable
import pickle
from bb_opt.src.hsic import (
    dimwise_mixrq_kernels,
    dimwise_mixrbf_kernels,
    precompute_batch_hsic_stats,
    compute_point_hsics,
    total_hsic,
)
import bb_opt.src.hsic as hsic
from bb_opt.src.knn_mi import estimate_mi
from bb_opt.src.utils import get_path


N_HIDDEN = 100
NON_LINEARITY = "ReLU"
SVI = None

InputType = torch.Tensor
LabelType = torch.Tensor
ModelType = Callable[[InputType], LabelType]


def ExtendedGamma(shape, scale, ndimension=0):
    device = "cuda" if scale.is_cuda else "cpu"
    beta = 1 / scale
    sign = pyro.sample("signX", Bernoulli(torch.tensor(0.5, device=device)))
    sign = sign * 2 - 1
    return sign * pyro.sample("egX", Gamma(shape, beta).independent(ndimension))


def normal_priors(
    model: torch.nn.Module, mean: float = 0., std: float = 1.
) -> Dict[str, TorchDistribution]:
    priors = {}
    for name, param in model.named_parameters():
        priors[name] = Normal(
            torch.full_like(param, fill_value=mean),
            torch.full_like(param, fill_value=std),
        ).independent(param.ndimension())

    return priors


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


def normal_variationals(
    model: torch.nn.Module, mean: float = 0., std: float = 1.
) -> Dict[str, TorchDistribution]:
    variational_dists = {}
    for name, param in model.named_parameters():
        location = pyro.param(f"g_{name}_location", torch.randn_like(param) + mean)
        log_scale = pyro.param(
            f"g_{name}_log_scale", torch.randn_like(param) + np.log(np.exp(std) - 1)
        )
        variational_dists[name] = Normal(
            location, torch.nn.Softplus()(log_scale)
        ).independent(param.ndimension())
    return variational_dists


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


def make_bnn_model(
    model: torch.nn.Module,
    priors: Callable[[], Dict[str, TorchDistribution]],
    batch_size: int = 128,
) -> Callable:
    def bnn_model(inputs, labels):
        bnn = pyro.random_module("bnn", model, priors())
        nn_sample = bnn()
        nn_sample.train()  # train mode on
        with pyro.iarange("i", len(inputs), subsample_size=batch_size) as i:
            pred = nn_sample(inputs[i]).squeeze()
            pyro.sample(
                "obs", Normal(pred, torch.ones_like(pred)), obs=labels[i].squeeze()
            )

    return bnn_model


def make_guide(
    model: torch.nn.Module,
    variational_dists: Callable[[], Dict[str, TorchDistribution]],
) -> Callable:
    def guide(inputs=None, labels=None):
        bnn = pyro.random_module("bnn", model, variational_dists())
        nn_sample = bnn()
        nn_sample.train()
        return nn_sample

    return guide


def optimize(
    # TODO: how to do the typing for inputs to `get_model`? needs batch_size and device somewhere
    get_model: Callable[..., ModelType],
    # TODO: need a similar typing update for acquisition (some required args, can be any other optional ones too)
    acquisition_func: Callable[
        [ModelType, InputType, LabelType, Set[int], int], Iterable[int]
    ],
    train_model: Callable[[ModelType, InputType, LabelType], None],
    inputs: Union[torch.Tensor, np.ndarray],
    labels: Union[pd.Series, np.ndarray],
    top_k_percent: int = 1,
    batch_size: int = 256,
    n_epochs: int = 2,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    exp: Optional = None,
    acquisition_args: Optional[dict] = None,
    n_batches: int = -1,
    retrain_every: int = 1,
    partial_train_steps: int = 10,
    partial_train_func: Optional[Callable] = None,
    save_key: str = "",
) -> np.ndarray:

    acquisition_args = acquisition_args or {}
    n_batches = n_batches if n_batches != -1 else int(np.ceil(len(labels) / batch_size))

    if isinstance(labels, pd.Series):
        labels = labels.values

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs).float()

    if device and inputs.device != device:
        inputs = inputs.to(device)

    n_top_k_percent = int(top_k_percent / 100 * len(labels))
    best_idx = set(labels.argsort()[-n_top_k_percent:])
    sum_best = np.sort(labels)[-n_top_k_percent:].sum()
    best_value = np.sort(labels)[-1]

    labels = torch.tensor(labels).float()
    if device and labels.device != device:
        labels = labels.to(device)

    try:
        model = get_model(
            n_inputs=inputs.shape[1], batch_size=batch_size, device=device
        )
        sampled_idx = set()
        fraction_best_sampled = []

        for step in range(n_batches):
            acquire_samples = acquisition_func(
                model,
                inputs,
                labels[list(sampled_idx)],
                sampled_idx,
                batch_size,
                exp=exp,
                **acquisition_args,
            )
            sampled_idx.update(acquire_samples)
            fraction_best = len(best_idx.intersection(sampled_idx)) / len(best_idx)
            fraction_best_sampled.append(fraction_best)

            if verbose:
                print(f"{step} {fraction_best:.3f}")

            if exp:
                sampled_labels = labels[list(sampled_idx)]
                sampled_sum_best = (
                    sampled_labels.sort()[0][-n_top_k_percent:].sum().item()
                )
                sampled_best_value = sampled_labels.max().item()

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
                train_model(
                    model,
                    inputs[list(sampled_idx)],
                    labels[list(sampled_idx)],
                    n_epochs,
                    batch_size,
                )
            else:
                if partial_train_func:
                    partial_train_func(
                        model,
                        inputs[list(acquire_samples)],
                        labels[list(acquire_samples)],
                        partial_train_steps,
                    )

        assert (len(sampled_idx) == min(len(labels), batch_size * n_batches)) or (
            fraction_best == 1.0
        )

    except KeyboardInterrupt:
        pass

    if save_key:
        save_fname = get_path(__file__, "..", "..", "models", save_key)
        pyro.get_param_store().save(f"{save_fname}.params")
        optimizer.save(f"{save_fname}.opt")

    return model, SVI


def get_model_uniform(*args, **kwargs):
    pass


def acquire_batch_uniform(
    model: ModelType,
    inputs: InputType,
    sampled_labels: LabelType,
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
    model: ModelType,
    inputs: InputType,
    sampled_labels: LabelType,
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


def get_model_bnn(
    n_inputs: int = 512,
    batch_size: int = 200,
    prior_mean: float = 0,
    prior_std: float = 1.0,
    device=None,
):
    device = device or "cpu"
    model = nn.Sequential(
        nn.Linear(n_inputs, N_HIDDEN),
        getattr(nn, NON_LINEARITY)(),
        nn.Linear(N_HIDDEN, 1),
    ).to(device)

    priors = lambda: normal_priors(model, prior_mean, prior_std)
    variational_dists = lambda: normal_variationals(model, prior_mean, prior_std)

    bnn_model = make_bnn_model(model, priors, batch_size=batch_size)
    guide = make_guide(model, variational_dists)
    return bnn_model, guide


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


def acquire_batch_bnn_greedy(
    model,
    inputs: InputType,
    sampled_labels: LabelType,
    sampled_idx: Set[int],
    batch_size: int,
    n_bnn_samples: int = 50,
    exp: Optional = None,
) -> List[int]:
    bnn_model, guide = model
    preds = bnn_predict(guide, inputs, n_samples=n_bnn_samples)
    preds = preds.mean(axis=0)
    sorted_idx = np.argsort(preds)
    sorted_idx = [i for i in sorted_idx if i not in sampled_idx]
    acquire_samples = sorted_idx[-batch_size:]  # sorted smallest first
    return acquire_samples


def acquire_batch_pdts(
    model,
    inputs: InputType,
    sampled_labels: LabelType,
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
    inputs: InputType,
    sampled_labels: LabelType,
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
    inputs: InputType,
    sampled_labels: LabelType,
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
    inputs: InputType,
    sampled_labels: LabelType,
    sampled_idx: Set[int],
    batch_size: int,
    n_points_parallel: int = 100,
    exp: Optional = None,
    kernel: Optional[Callable] = None,
    mi_estimator: str = "HSIC",
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
    )


def acquire_batch_es(
    model,
    inputs: InputType,
    sampled_labels: LabelType,
    sampled_idx: Set[int],
    batch_size: int,
    n_points_parallel: int = 100,
    exp: Optional = None,
    kernel: Optional[Callable] = None,
    mi_estimator: str = "HSIC",
    max_value_es: bool = False,
) -> List[int]:

    acquirable_idx = set(range(len(inputs))).difference(sampled_idx)

    n_preds = 250
    bnn_model, guide = model

    with torch.no_grad():
        preds = torch.stack([guide()(inputs).squeeze() for _ in range(n_preds)])

    if max_value_es:
        max_dist = preds.max(dim=1)[0]
    else:
        max_pred_idx = preds.argmax(dim=1)
        max_dist = inputs[max_pred_idx]

    max_dist = max_dist.unsqueeze(1)
    acquirable_idx = list(acquirable_idx)
    batch = []

    if mi_estimator == "HSIC":
        max_batch_dist = max_dist

        for _ in range(batch_size):
            all_mi = total_hsic_batched(max_batch_dist, preds, kernel, acquirable_idx)
            best_idx = acquirable_idx[all_mi.argmax().item()]
            acquirable_idx.remove(best_idx)
            batch.append(best_idx)
            max_batch_dist = torch.cat(
                (max_batch_dist, preds[:, best_idx : best_idx + 1]), dim=-1
            )

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
            mode_count = (max_idx == max_idx.mode()[0]).sum().item()
            selected_count = (max_idx == batch[0]).sum().item()

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


def hsic_mves_loss(
        params,
        X : torch.tensor, # (num_samples, ack_batch_size)
        opt_values_kernel_matrix : torch.tensor,
        kernel_fn,
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

    new_batch_matrix = kernel_fn(X, X) # return is of shape (n=num_samples, n, 1)
    kernels = torch.cat([X, opt_values_kernel_matrix], dim=-1)

    return -hsic.total_hsic(kernels)

def acquire_batch_via_grad_mves(
        params,
        model_ensemble : nn.Module,
        opt_values : torch.tensor,
        input_shape : List[int],
        seed : torch.tensor = None,
        device : str = "cuda",
)-> torch.tensor:

    ack_batch_size = params.ack_batch_size
    if seed is None:
        input_tensor = torch.randn([ack_batch_size]+input_shape, device=device, requires_grad=True)
    else:
        assert seed.shape[0] == ack_batch_size
        input_tensor = torch.tensor(seed, device=device, requires_grad=True)

    optim = torch.optim.Adam([input_tensor], lr=params.input_opt_lr)
    kernel_fn = getattr(hsic, 'two_vec_' + params.mves_kernel_fn)
    opt_kernel_matrix = kernel_fn(opt_values, opt_values) # return is of shape (n=num_samples, n, 1)

    for step_iter in params.input_opt_num_iter:
        preds = model_ensemble(input_tensor) # output should be (num_samples, ack_batch_size)
        loss = hsic_mves_loss(params, input_tensor, opt_kernel_matrix, kernel_fn)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

    return input_tensor


def acquire_batch_via_grad_ei(
        params,
        model_ensemble : nn.Module,
        input_shape : List[int],
        seed : torch.tensor = None,
        device : str = "cuda",
)-> torch.tensor:

    ack_batch_size = params.ack_batch_size
    if seed is None:
        input_tensor = torch.randn([ack_batch_size]+input_shape, device=device, requires_grad=True)
    else:
        assert seed.shape[0] == ack_batch_size
        input_tensor = torch.tensor(seed, device=device, requires_grad=True)

    optim = torch.optim.Adam([input_tensor], lr=params.input_opt_lr)
    kernel_fn = getattr(hsic, 'two_vec_' + params.mves_kernel_fn)
    opt_kernel_matrix = kernel_fn(opt_values, opt_values) # return is of shape (n=num_samples, n, 1)

    for step_iter in params.input_opt_num_iter:
        preds = model_ensemble(input_tensor) # output should be (num_samples, ack_batch_size)
        loss = -torch.mean(preds)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

    return input_tensor


def optimize_model_input(
        params,
        input_shape,
        model_ensemble,
        seed=None,
):
    ack_num_model_samples = params.ack_num_model_samples
    if seed is None:
        input_tensor = torch.randn([ack_num_model_samples]+input_shape, device=device, requires_grad=True)
    else:
        assert seed.shape[0] == ack_num_model_samples
        input_tensor = torch.tensor(seed, device=device, requires_grad=True).unsqueeze(0)

    optim = torch.optim.Adam([input_tensor], lr=params.input_opt_lr)
    for step_iter in params.input_opt_num_iter:
        preds = model_ensemble(input_tensor)
        loss = -torch.sum(pred)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

    return input_tensor

def acquire_batch_pi(
    model,
    inputs: InputType,
    sampled_labels: LabelType,
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


def train_model_bnn(model, inputs, labels, n_epochs: int, batch_size: int):
    global SVI
    bnn_model, guide = model
    optimizer = pyro.optim.Adam({})
    pyro.clear_param_store()
    svi = pyro.infer.SVI(bnn_model, guide, optimizer, loss=pyro.infer.Trace_ELBO())
    if n_epochs == -1:
        n_steps = 10000000000000
    else:
        n_steps = int(len(inputs) / batch_size * n_epochs)
    SVI = svi
    train(svi, n_steps, inputs, labels)


def partial_train_model_bnn(model, inputs, labels, n_steps: int):
    global SVI
    if SVI is None:
        bnn_model, guide = model
        optimizer = pyro.optim.Adam({})
        svi = pyro.infer.SVI(bnn_model, guide, optimizer, loss=pyro.infer.Trace_ELBO())
        SVI = svi
    train(SVI, n_steps, inputs, labels)


def get_early_stopping(
    max_patience: int, threshold: float = 1.0
) -> Callable[[float], bool]:
    old_min = float("inf")
    patience = max_patience

    def early_stopping(metric: float) -> bool:
        nonlocal old_min
        nonlocal patience
        nonlocal max_patience
        if metric < threshold * old_min:
            old_min = metric
            patience = max_patience
        else:
            patience -= 1

        if patience == 0:
            return True
        return False

    return early_stopping


def train(
    svi,
    n_steps: int,
    inputs,
    labels,
    verbose: bool = False,
    max_patience: int = 3,
    n_steps_early_stopping: int = 300,
):
    losses = []
    early_stopping = get_early_stopping(max_patience)

    try:
        for step in range(n_steps):
            loss = svi.step(inputs, labels)
            losses.append(loss)
            if step % n_steps_early_stopping == 0:
                if verbose:
                    print(f"[S{step:04}] loss: {loss:,.0f}")

                stop_now = early_stopping(min(losses[-n_steps_early_stopping:]))
                if stop_now:
                    break
    except KeyboardInterrupt:
        pass
    return losses


def bnn_predict(guide: Callable, inputs: torch.Tensor, n_samples: int) -> np.ndarray:
    preds = []

    with torch.no_grad():
        for _ in range(n_samples):
            nn_sample = guide()
            nn_sample.eval()
            preds.append(nn_sample(inputs).cpu().squeeze().numpy())
    return np.array(preds)


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
