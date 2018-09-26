"""
Deep ensemble model from [this paper]().

Only a simple MLP is supported at present, but it should be easy to ensemble other
`torch.nn.Module`s by using them instead of `NN` in `NNEnsemble`.

This implementation also only supports training all models in the ensemble at once
on the same GPU; modifications would be needed to use this for larger models.
"""

import torch
from torch.nn import Linear, ReLU, Softplus
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Optional, Dict


class NN(torch.nn.Module):
    """
    Single-layer MLP that predicts a Gaussian for each point.
    """

    def __init__(self, n_inputs: int, n_hidden: int, min_variance: float = 1e-5):
        super().__init__()
        self.hidden = Linear(n_inputs, n_hidden)
        self.output = Linear(n_hidden, 2)
        self.non_linearity = ReLU()
        self.softplus = Softplus()
        self.min_variance = min_variance

    def forward(self, x):
        hidden = self.non_linearity(self.hidden(x))
        output = self.output(hidden)
        mean = output[:, 0]
        variance = self.softplus(output[:, 1]) + self.min_variance
        return mean, variance


class NNEnsemble(torch.nn.Module):
    """
    Ensemble of `NN`s, trained individually but whose predictions can be combined.
    Adversarial training will be used if `adversarial_epsilon` is not `None`.

    Note that if training your model takes a lot of GPU resources, it would be
    better to train the individual models on separate GPUs in parallel and then
    save them and load the weights into an ensemble afterwards for evaluation.
    """

    def __init__(
        self,
        n_models: int,
        n_inputs: int,
        n_hidden: int,
        min_variance: float = 1e-5,
        adversarial_epsilon: Optional = None,
    ):
        super().__init__()
        self.n_models = n_models
        self.models = torch.nn.ModuleList(
            [NN(n_inputs, n_hidden, min_variance) for _ in range(n_models)]
        )
        self.adversarial_epsilon = adversarial_epsilon

    def forward(self, x, y=None, optimizer=None, individual_predictions: bool = True):
        if y is not None and self.adversarial_epsilon is not None:
            x.requires_grad_()
            means, variances = self(x)
            negative_log_likelihood = compute_negative_log_likelihood(
                y, means, variances
            )

            grad = torch.autograd.grad(
                negative_log_likelihood, x, retain_graph=optimizer is not None
            )[0]
            x = x.detach() + self.adversarial_epsilon * torch.sign(grad)

            if optimizer:
                # then do a backward pass on x as well as returning the prediction
                # for x_adv to do a pass on that
                negative_log_likelihood.backward()
                optimizer.step()
                optimizer.zero_grad()

        means, variances = list(zip(*[self.models[i](x) for i in range(self.n_models)]))
        means, variances = torch.stack(means), torch.stack(variances)

        if individual_predictions:
            return means, variances

        return combine_means_variances(means, variances)


def combine_means_variances(
    means: torch.Tensor, variances: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Combine individual means/variances of Gaussian mixture into the mixture mean/variance.

    Note that we assume equal weighting between mixture components.
    The mean of the mixture is just the average of the individual Gaussians means:
      $$\mu^* = \frac 1 M \sum_{i=1}^M \mu_i$$
    The variance is
      $${\sigma^*}^2 = \frac 1 M \sum_{i=1}^M (\sigma^2_i + \mu_i^2) - {\mu^*}^2$$

    :param means: n_models x n_inputs
    :param variances: n_models x n_inputs
    :returns: (mean, variance) where each is of shape n_inputs
    """
    mean = means.mean(dim=0)
    variance = (variances + means ** 2).mean(dim=0) - mean ** 2
    return mean, variance


def compute_negative_log_likelihood(labels, means, variances, return_mse: bool = False):
    mse = (labels - means) ** 2
    negative_log_likelihood = 0.5 * (torch.log(variances) + mse / variances)
    negative_log_likelihood = negative_log_likelihood.mean(dim=-1).sum()

    if return_mse:
        mse = mse.mean(dim=-1).sum()
        return negative_log_likelihood, mse
    return negative_log_likelihood


def compute_weighted_nll(labels, means, variances, return_mse: bool = False):
    mse = (labels - means) ** 2
    negative_log_likelihood = 0.5 * (torch.log(variances) + mse / variances)
    negative_log_likelihood = negative_log_likelihood.mean(dim=-1)
    negative_log_likelihood = (labels * negative_log_likelihood).sum()

    if return_mse:
        mse = mse.mean(dim=-1).sum()
        return negative_log_likelihood, mse
    return negative_log_likelihood


def get_model_deep_ensemble(
    n_inputs: int,
    batch_size: int = 200,
    n_models: int = 5,
    n_hidden: int = 100,
    adversarial_epsilon: Optional = None,
    device=None,
):
    device = device or "cpu"
    model = NNEnsemble(
        n_models, n_inputs, n_hidden, adversarial_epsilon=adversarial_epsilon
    ).to(device)
    return model


def train_model_deep_ensemble(
    model, inputs, labels, n_epochs, batch_size, optimizer_kwargs: Optional[Dict] = None
):
    optimizer_kwargs = optimizer_kwargs or {}
    data = TensorDataset(inputs, labels)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

    model.train()
    for epoch in range(n_epochs):
        for batch in loader:
            inputs, labels = batch
            optimizer.zero_grad()

            means, variances = model(inputs)

            negative_log_likelihood = compute_negative_log_likelihood(
                labels, means, variances
            )
            negative_log_likelihood.backward()
            optimizer.step()
    model.eval()
