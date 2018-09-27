import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as get_fingerprint
import torch
from tqdm import tnrange
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
    total_hsic_batched,
)
import bb_opt.src.hsic as hsic
from bb_opt.src.knn_mi import estimate_mi
from bb_opt.src.utils import get_path, save_checkpoint


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
    acquisition_func_greedy: Optional[Callable] = None,
) -> np.ndarray:

    acquisition_args = acquisition_args or {}
    acquisition_func_greedy = acquisition_func_greedy or acquisition_func
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
            if step == 0:
                acquire_samples = acquire_batch_uniform(
                    model,
                    inputs,
                    labels[list(sampled_idx)],
                    sampled_idx,
                    batch_size,
                    exp=exp,
                )
            else:
                acquire_samples = acquisition_func(
                    model,
                    inputs,
                    labels[list(sampled_idx)],
                    sampled_idx,
                    batch_size,
                    exp=exp,
                    **acquisition_args,
                )

            acquire_samples_greedy = acquisition_func_greedy(
                model,
                inputs,
                labels[list(sampled_idx)],
                sampled_idx,
                batch_size,
                exp=exp,
                **acquisition_args,
            )

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

            if save_key:
                save_fname = get_path(
                    __file__, "..", "..", "models_nathan", save_key, step
                )

                if "de" in save_key:
                    save_checkpoint(f"{save_fname}.pth", model)
                elif "bnn" in save_key:
                    pyro.get_param_store().save(f"{save_fname}.params")

        assert (len(sampled_idx) == min(len(labels), batch_size * n_batches)) or (
            fraction_best == 1.0
        )
    except KeyboardInterrupt:
        pass

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


def acquire_batch_ei(
    model,
    inputs: InputType,
    sampled_labels: LabelType,
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
    inputs: InputType,
    sampled_labels: LabelType,
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


def hsic_mves_loss(
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

    return -hsic.total_hsic(kernels)


def acquire_batch_via_grad_mves(
    params,
    model_ensemble: Callable[[torch.tensor], torch.tensor],
    input_shape: List[int],
    opt_values: torch.tensor, # (num_samples, preds)
    ack_batch_size,
    do_mean=False,
    seed: torch.tensor = None,
) -> torch.tensor:

    print('mves: ack_batch_size', ack_batch_size)
    if seed is None:
        input_tensor = torch.randn([ack_batch_size] + input_shape, device=params.device, requires_grad=True)
    else:
        assert seed.shape[0] == ack_batch_size
        input_tensor = torch.tensor(seed, device=device, requires_grad=True)

    optim = torch.optim.Adam([input_tensor], lr=params.input_opt_lr)
    kernel_fn = getattr(hsic, "two_vec_" + params.mves_kernel_fn)
    opt_kernel_matrix = kernel_fn(opt_values, opt_values, do_mean=do_mean)  # shape (n=num_samples, n, 1)
    progress = tnrange(params.batch_opt_num_iter)

    for step_iter in progress:
        preds = model_ensemble(input_tensor, resize_at_end=True) # (num_samples, ack_batch_size)
        assert opt_kernel_matrix.shape[0] == preds.shape[0], str(opt_kernel_matrix.shape) + "[0] == " + str(preds.shape) + "[0]"
        loss = hsic_mves_loss(preds, opt_kernel_matrix, kernel_fn, do_mean)
        postfix = {'loss' : loss.item()}
        progress.set_postfix(postfix)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

    return input_tensor.detach()

def acquire_batch_mves_sid(
        params,
        opt_values : torch.tensor,
        preds : torch.tensor, # (num_samples, num_candidate_points)
        skip_idx,
        mves_compute_batch_size,
        do_mean=False,
        device : str = "cuda",
)-> torch.tensor:

    max_pred_idx = set(preds.argmax(1).detach().cpu().numpy())

    print('num_max_pred_idx:', len(max_pred_idx))

    num_candidate_points = preds.shape[1]
    num_samples = preds.shape[0]

    opt_kernel_matrix = kernel_fn(opt_values, opt_values, do_mean=do_mean)  # shape (n=num_samples, n, 1)
    assert opt_values_kernel_matrix.ndimension() == 3
    assert opt_values_kernel_matrix.shape[0] == num_samples
    assert opt_values_kernel_matrix.shape[1] == num_samples
    assert opt_values_kernel_matrix.shape[-1] == 1
    opt_kernel_matrix = opt_kernel_matrix.permute([2, 0, 1]).unsqueeze(-1) # (1, n, n, 1)

    batch_idx = set()
    remaining_idx_set = set(range(num_candidate_points))
    batch_dist_matrix = None # (1, num_samples, num_samples, 1)

    regular_batch_opt_kernel_matrix = opt_kernel_matrix.repeat([mves_compute_batch_size, 1, 1, 1])
    while len(batch_idx) < ack_batch_size:
        best_idx = None
        best_idx_dist_matrix = None
        best_hsic = None

        num_remaining = len(remaining_idx)
        num_batches = num_remaining // mves_compute_batch_size
        remaining_idx = torch.tensor(list(remaining_idx_set), device=device)

        for bi in range(num_batches):
            bs = bi*mves_compute_batch_size
            be = min((bi*1)*mves_compute_batch_size, num_remaining)
            cur_batch_size = bs-be+1
            idx = remaining_idx[bs:be]

            pred = candidate_points_preds[idx, :] # (mves_compute_batch_size, num_samples)
            dist_matrix = hsic.sqdist(pred) # (num_samples, num_samples, cur_batch_size)
            assert dist_matrix.ndimension() == 3
            assert dist_matrix.shape[2] == cur_batch_size
            assert dist_matrix.shape[0] == num_samples
            assert dist_matrix.shape[1] == num_samples

            dist_matrix = dist_matrix.permute([2, 0, 1]).unsqueeze(-1) # (cur_batch_size, num_samples, num_samples, 1)
            if batch_dist_matrix is not None:
                dist_matrix += batch_dist_matrix

            if cur_batch_size == mves_compute_batch_size:
                kernels = torch.concat([dist_matrix, regular_batch_opt_kernel_matrix], dim=-1)
            else:
                last_batch_opt_kernel_matrix = opt_kernel_matrix.repeat([cur_batch_size, 1, 1, 1])
                kernels = torch.concat([dist_matrix, last_batch_opt_kernel_matrix], dim=-1)

            total_hsic = hsic.total_hsic_paralle(kernels)
            assert total_hsic.ndimension() == 1
            sorted_idx = total_hsic.cpu().numpy().argsort()

            best_cur_idx = None
            for idx in sorted_idx[::-1]:
                if idx not in skip_idx:
                    best_cur_idx = idx
                    break

            if not (best_cur_idx is None) and (best_idx is None or best_hsic < total_hsic):
                best_idx = idx[best_cur_idx]
                best_idx_dist_matrix = dist_matrix[best_cur_idx:best_cur_idx+1]
                best_hsic = total_hsic

        assert best_hsic is not None
        assert best_idx_dist_matrix is not None
        assert best_idx is not None

        batch_dist_matrix = best_idx_dist_matrix
        remaining_idx_set.remove(best_idx)
        batch_idx.update(best_idx)

    return batch_idx


def acquire_batch_via_grad_ei(
    params,
    model_ensemble: Callable[[torch.tensor], torch.tensor],
    input_shape: List[int],
    seed: torch.tensor = None,
) -> torch.tensor:

    ack_batch_size = params.ack_batch_size
    if seed is None:
        input_tensor = torch.randn(
            [ack_batch_size] + input_shape, device=params.device, requires_grad=True
        )
    else:
        assert seed.shape[0] == ack_batch_size
        input_tensor = torch.tensor(seed, device=params.device, requires_grad=True)

    optim = torch.optim.Adam([input_tensor], lr=params.batch_opt_lr)
    kernel_fn = getattr(hsic, 'two_vec_' + params.mves_kernel_fn)

    for step_iter in range(params.batch_opt_num_iter):
        preds = model_ensemble(input_tensor, resize_at_end=True) # (num_samples, ack_batch_size)
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
    hsic_diversity_lambda=0.
):
    if seed is None:
        input_tensor = torch.randn([num_points_to_optimize] + input_shape, device=params.device, requires_grad=True)
    else:
        assert seed.ndimension() == 2
        assert seed.shape[0] == num_points_to_optimize
        input_tensor = torch.tensor(seed, device=params.device, requires_grad=True)

    optim = torch.optim.Adam([input_tensor], lr=params.input_opt_lr)
    progress = tnrange(params.input_opt_num_iter)
    for step_iter in progress:
        preds = model_ensemble(input_tensor, resize_at_end=True) # (num_samples, ack_batch_size)

        loss = -torch.mean(preds)

        postfix = {'normal_loss' : loss.item()}
        if hsic_diversity_lambda > 1e-9:
            kernels = hsic.dimwise_mixrq_kernels(preds)
            total_hsic = hsic.total_hsic(kernels)
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
):
    assert num_points_to_optimize > 1
    if seed is None:
        input_tensor = torch.randn([num_points_to_optimize] + input_shape, device=params.device, requires_grad=True)
    else:
        assert seed.ndimension() == 2
        assert seed.shape[0] == num_points_to_optimize
        input_tensor = torch.tensor(seed, device=params.device, requires_grad=True)

    optim = torch.optim.Adam([input_tensor], lr=params.input_opt_lr)
    progress = tnrange(params.input_opt_num_iter)
    for step_iter in progress:
        preds = model_ensemble(input_tensor, all_pairs=False) # (num_samples,)
        assert preds.shape[0] == num_points_to_optimize

        loss = -torch.mean(preds)

        postfix = {'normal_loss' : loss.item()}
        optim.zero_grad()
        loss.backward()
        optim.step()
        progress.set_postfix(postfix)

    return input_tensor.detach(), preds.detach()


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
