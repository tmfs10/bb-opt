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
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Tuple, Optional, Callable, Dict, Set, List, Any, Iterable
import pickle
from bb_opt.src.hsic import (
    dimwise_mixrq_kernels,
    precompute_batch_hsic_stats,
    compute_point_hsics,
    total_hsic,
)


N_HIDDEN = 100
NON_LINEARITY = "ReLU"

InputType = torch.Tensor
LabelType = Union[torch.Tensor, np.ndarray]
ModelType = Callable[[InputType], LabelType]


def normal_priors(
    model: torch.nn.Module, mean: float = 0, std: float = 1
) -> Dict[str, pyro.distributions.TorchDistribution]:
    priors = {}
    for name, param in model.named_parameters():
        priors[name] = pyro.distributions.Normal(
            torch.ones_like(param) * mean, torch.ones_like(param) * std
        ).independent(param.ndimension())

    return priors


class SpikeSlabNormal(pyro.distributions.TorchDistribution):
    def __init__(self, param, std1: float = 0.03, std2: float = 2, alpha: float = 0.7):
        super().__init__(event_shape=param.shape)
        self.normal1 = pyro.distributions.Normal(
            torch.zeros_like(param), torch.ones_like(param) * std1
        ).independent(param.ndimension())

        self.normal2 = pyro.distributions.Normal(
            torch.zeros_like(param), torch.ones_like(param) * std2
        ).independent(param.ndimension())

        self.alpha = torch.tensor(alpha).to(param.device)
        self.bernoulli = pyro.distributions.Bernoulli(self.alpha)

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
    model: torch.nn.Module, std1: float = 0.03, std2: float = 2, alpha: float = 0.7
) -> Dict[str, pyro.distributions.TorchDistribution]:
    priors = {}
    for name, param in model.named_parameters():
        priors[name] = SpikeSlabNormal(param, std1, std2, alpha)

    return priors


def normal_variationals(
    model: torch.nn.Module, mean: float = 0, std: float = 1
) -> Dict[str, pyro.distributions.TorchDistribution]:
    variational_dists = {}
    for name, param in model.named_parameters():
        location = pyro.param(f"g_{name}_location", torch.randn_like(param) + mean)
        log_scale = pyro.param(
            f"g_{name}_log_scale", torch.randn_like(param) + np.log(np.exp(std) - 1)
        )
        variational_dists[name] = pyro.distributions.Normal(
            location, torch.nn.Softplus()(log_scale)
        ).independent(param.ndimension())
    return variational_dists


def make_bnn_model(
    model: torch.nn.Module,
    priors: Callable[[], Dict[str, pyro.distributions.TorchDistribution]],
    batch_size: int = 128,
) -> Callable:
    def bnn_model(inputs, labels):
        bnn = pyro.random_module("bnn", model, priors())
        nn_sample = bnn()
        nn_sample.train()  # train mode on
        with pyro.iarange("i", len(inputs), subsample_size=batch_size) as i:
            pred = nn_sample(inputs[i]).squeeze()
            pyro.sample(
                "obs",
                pyro.distributions.Normal(pred, torch.ones_like(pred)),
                obs=labels[i].squeeze(),
            )

    return bnn_model


def make_guide(
    model: torch.nn.Module,
    variational_dists: Callable[[], Dict[str, pyro.distributions.TorchDistribution]],
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
    acquisition_func: Callable[[ModelType, InputType, Set[int], int], Iterable[int]],
    train_model: Callable[[ModelType, InputType, LabelType], None],
    inputs: Union[torch.Tensor, np.ndarray],
    labels: Union[pd.Series, np.ndarray],
    top_k_percent: int = 1,
    n_repeats: int = 1,
    batch_size: int = 256,
    n_epochs: int = 2,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    exp: Optional = None,
) -> Tuple[np.ndarray, np.ndarray]:

    if isinstance(labels, pd.Series):
        labels = labels.values

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs).float()

    if device and inputs.device != device:
        inputs = inputs.to(device)

    n_top_k_percent = int(top_k_percent / 100 * len(labels))
    best_idx = set(labels.argsort()[-n_top_k_percent:])
    labels = torch.tensor(labels).float()
    if device and labels.device != device:
        labels = labels.to(device)

    all_fraction_best_sampled = []
    try:
        for _ in range(n_repeats):
            model = get_model(batch_size=batch_size, device=device)
            sampled_idx = set()
            fraction_best_sampled = []

            for step in range(int(np.ceil(len(labels) / batch_size))):
                acquire_samples = acquisition_func(
                    model, inputs, sampled_idx, batch_size, exp=exp
                )
                sampled_idx.update(acquire_samples)
                fraction_best = len(best_idx.intersection(sampled_idx)) / len(best_idx)
                fraction_best_sampled.append(fraction_best)

                if verbose:
                    print(step, fraction_best)

                if exp:
                    exp.log_metric("fraction_best", fraction_best, step)

                train_model(
                    model,
                    inputs[list(sampled_idx)],
                    labels[list(sampled_idx)],
                    n_epochs,
                    batch_size,
                )

            assert len(sampled_idx) == len(labels)

            all_fraction_best_sampled.append(fraction_best_sampled)
    except KeyboardInterrupt:
        pass

    fraction_best_sampled = np.array(all_fraction_best_sampled)
    mean_fraction_best = fraction_best_sampled.mean(axis=0)
    std_fraction_best = fraction_best_sampled.std(axis=0)
    return mean_fraction_best, std_fraction_best


def get_model_uniform(*args, **kwargs):
    pass


def acquire_batch_uniform(
    model: ModelType, inputs: InputType, sampled_idx: Set[int], batch_size: int
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
    model: ModelType, inputs: InputType, sampled_idx: Set[int], batch_size: int
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
    prior_std: float = 0.05,
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


def acquire_batch_bnn_greedy(
    model,
    inputs: InputType,
    sampled_idx: Set[int],
    batch_size: int,
    n_bnn_samples: int = 50,
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
    sampled_idx: Set[int],
    batch_size: int,
    exp: Optional = None,
) -> Set[int]:
    bnn_model, guide = model
    # preds = bnn_predict(guide, inputs, n_samples=batch_size)

    with torch.no_grad():
        preds = [
            torch.unsqueeze(guide()(inputs).squeeze(), 0) for _ in range(batch_size)
        ]
    preds = torch.cat(preds)

    # TODO: sorting makes this O(mn log n) when it should just be O(mn)
    # however, we can't just take the max from each sampled NN in case
    # it's already been acquired or is acquired by another of the NN samples
    # if we don't even predict on the already acquired samples, then we
    # could just take the m max points from each NN sample and use the largest
    # of those that isn't already selected by another NN sample
    # what would the runtime for this be? if m is O(n), would it be any
    # better than O(n log n)?
    sorted_idx = np.argsort(preds).numpy()

    acquire_samples = set()
    for row in sorted_idx:
        for idx in row[::-1]:  # largest last so reverse
            # avoid acquiring the same point from multiple nn samples
            if idx in acquire_samples or idx in sampled_idx:
                continue

            acquire_samples.add(idx)
            break

    hsic = total_hsic(dimwise_mixrq_kernels(preds[:, list(acquire_samples)]))
    exp.log_metric("hsic_mean", hsic, step=len(sampled_idx))

    return acquire_samples


def acquire_batch_hsic(
    model,
    inputs: InputType,
    sampled_idx: Set[int],
    batch_size: int,
    n_points_parallel: int = 100,
    exp: Optional = None,
) -> List[int]:
    hsic_coeff = 150
    n_preds = 250
    bnn_model, guide = model

    with torch.no_grad():
        preds = [torch.unsqueeze(guide()(inputs).squeeze(), 0) for _ in range(n_preds)]
    preds = torch.cat(preds)

    mean = preds.mean(dim=0)
    std = preds.std(dim=0)
    metric = mean / std

    acquirable_idx = set(range(len(metric))).difference(sampled_idx)

    best_metric = -float("inf")
    best_idx = None

    for idx in acquirable_idx:
        if metric[idx] > best_metric:
            best_metric = metric[idx]
            best_idx = idx

    batch = [best_idx]
    acquirable_idx.remove(best_idx)

    while len(batch) < batch_size:
        best_idx = None
        best_batch_metric = -float("inf")

        batch_stats = precompute_batch_hsic_stats(preds, batch)

        all_hsics = []

        for next_points in torch.tensor(list(acquirable_idx)).split(n_points_parallel):
            hsics = compute_point_hsics(preds, next_points, *batch_stats)
            idx = hsics.argmax()
            hsic = hsics[idx]
            idx = next_points[idx]

            all_hsics.append(hsics.cpu().numpy())

            batch_metric = metric[idx] - hsic_coeff * hsic

            if batch_metric > best_batch_metric:
                best_batch_metric = batch_metric
                best_idx = idx.item()
            else:
                print(best_metric, hsic_coeff, hsic, metric[idx])

        batch.append(best_idx)
        acquirable_idx.remove(best_idx)

        all_hsics = np.concatenate(all_hsics)
        exp.log_multiple_metrics(
            {"hsic_mean": np.mean(all_hsics), "hsic_std": np.std(all_hsics)},
            step=len(sampled_idx) + len(batch),
        )

        if not acquirable_idx:
            break
    return batch


def train_model_bnn(model, inputs, labels, n_epochs: int, batch_size: int):
    bnn_model, guide = model
    optimizer = pyro.optim.Adam({})
    pyro.clear_param_store()
    svi = pyro.infer.SVI(bnn_model, guide, optimizer, loss=pyro.infer.Trace_ELBO())
    if n_epochs == -1:
        n_steps = 10000000000000
    else:
        n_steps = int(len(inputs) / batch_size * n_epochs)
    train(svi, n_steps, inputs, labels)


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
    max_patience: int = 5,
    n_steps_early_stopping: int = 1000,
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
