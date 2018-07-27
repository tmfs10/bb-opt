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
from typing import Union, Tuple, Optional, Callable, Dict, Set, List, Any
from argparse import ArgumentParser
import pickle

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
    acquisition_func: Callable[[ModelType, InputType, Set[int], int], List[int]],
    train_model: Callable[[ModelType, InputType, LabelType], None],
    inputs: Union[torch.Tensor, np.ndarray],
    labels: pd.Series,
    top_k_percent: int = 1,
    n_repeats: int = 1,
    batch_size: int = 256,
    n_epochs: int = 2,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs).float()

    if device and inputs.device != device:
        inputs = inputs.to(device)

    n_top_k_percent = int(top_k_percent / 100 * len(labels))
    best_idx = set(labels.sort_values(ascending=False)[:n_top_k_percent].index)
    labels = torch.tensor(labels).float()
    if device and labels.device != device:
        labels = labels.to(device)

    all_fraction_best_sampled = []
    for i in range(n_repeats):
        model = get_model(batch_size=batch_size, device=device)
        sampled_idx = set()
        fraction_best_sampled = []

        for j in range(int(np.ceil(len(labels) / batch_size))):
            acquire_samples = acquisition_func(model, inputs, sampled_idx, batch_size)
            sampled_idx.update(acquire_samples)
            fraction_best_sampled.append(
                len(best_idx.intersection(sampled_idx)) / len(best_idx)
            )
            train_model(
                model,
                inputs[list(sampled_idx)],
                labels[list(sampled_idx)],
                n_epochs,
                batch_size,
            )

        assert len(sampled_idx) == len(labels)

        all_fraction_best_sampled.append(fraction_best_sampled)

    fraction_best_sampled = np.array(all_fraction_best_sampled)
    mean_fraction_best = fraction_best_sampled.mean(axis=0)
    std_fraction_best = fraction_best_sampled.std(axis=0)
    return mean_fraction_best, std_fraction_best


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
    model, inputs: InputType, sampled_idx: Set[int], batch_size
) -> List[int]:
    bnn_model, guide = model
    preds = bnn_predict(guide, inputs, n_samples=batch_size)
    sorted_idx = np.argsort(preds)

    acquire_samples = []
    for row in sorted_idx:
        for idx in row[::-1]:  # largest last so reverse
            # avoid acquiring the same point from multiple nn samples
            if idx in acquire_samples or idx in sampled_idx:
                continue

            acquire_samples.append(idx)
            break

    return acquire_samples


def train_model_bnn(model, inputs, labels, n_epochs: int, batch_size: int):
    bnn_model, guide = model
    optimizer = pyro.optim.Adam({})
    pyro.clear_param_store()
    svi = pyro.infer.SVI(bnn_model, guide, optimizer, loss=pyro.infer.Trace_ELBO())
    n_steps = int(len(inputs) / batch_size * n_epochs)
    train(svi, n_steps, inputs, labels)


def train(svi, n_steps: int, inputs, labels, verbose: bool = False):
    losses = []

    for step in range(n_steps):
        loss = svi.step(inputs, labels)
        losses.append(loss)
        if verbose and step % 500 == 0:
            print(f"[S{step:04}] loss: {loss:,.0f}")
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--device_num", type=int, default=0)
    parser.add_argument("-n", "--n_epochs", type=int, default=1)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv("../data/malaria.csv")
    bond_radius = 2
    n_bits = 512

    molecules = [MolFromSmiles(smile) for smile in data.smile]
    fingerprints = [
        get_fingerprint(mol, radius=bond_radius, nBits=n_bits) for mol in molecules
    ]
    fingerprints = np.array([[int(i) for i in fp.ToBitString()] for fp in fingerprints])

    fraction_best = {}  # {model_name: (mean, std)}
    n_repeats = 1
    top_k_percent = 1
    batch_size = 200

    mean, std = optimize(
        get_model_bnn,
        acquire_batch_pdts,
        train_model_bnn,
        fingerprints,
        data.ec50,
        top_k_percent,
        n_repeats,
        batch_size,
        args.n_epochs,
        device,
    )
    fraction_best["PDTS"] = (mean, std)

    rand = np.random.randint(100000000)
    with open(
        f"../plot_data/fraction_best_n_epochs_{args.n_epochs}_{rand}.pkl", "wb"
    ) as f:
        pickle.dump(fraction_best, f)
