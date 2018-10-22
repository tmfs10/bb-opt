import pickle
import numpy as np
from typing import TypeVar, Type, Optional, Callable, Union, Tuple, Any, Dict, List
import torch
from torch import nn
import pyro
from pyro.distributions import Normal, TorchDistribution
from bb_opt.src.bo_model import BOModel
from bb_opt.src.utils import get_early_stopping

_BNN = TypeVar("BNN", bound="BNN")


def normal_priors(
    model: nn.Module, mean: float = 0., std: float = 1.
) -> Dict[str, TorchDistribution]:
    priors = {}
    for name, param in model.named_parameters():
        priors[name] = Normal(
            torch.full_like(param, fill_value=mean),
            torch.full_like(param, fill_value=std),
        ).independent(param.ndimension())

    return priors


def normal_variationals(
    model: nn.Module, mean: float = 0., std: float = 1.
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


def make_bnn_model(
    model: nn.Module,
    priors: Callable[[], Dict[str, TorchDistribution]],
    batch_size: int = 128,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    use_cuda = next(model.parameters()).device.type == "cuda"

    def bnn_model(inputs: torch.Tensor, labels: torch.Tensor):
        bnn = pyro.random_module("bnn", model, priors())
        nn_sample = bnn()
        nn_sample.train()  # train mode on
        with pyro.iarange("batch_idx", len(inputs), batch_size, use_cuda=use_cuda) as i:
            pred = nn_sample(inputs[i]).squeeze()
            pyro.sample(
                "obs", Normal(pred, torch.ones_like(pred)), obs=labels[i].squeeze()
            )

    return bnn_model


def make_guide(
    model: nn.Module, variational_dists: Callable[[], Dict[str, TorchDistribution]]
) -> Callable[[Optional[torch.Tensor], Optional[torch.Tensor]], nn.Module]:
    def guide(
        inputs: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None
    ):
        bnn = pyro.random_module("bnn", model, variational_dists())
        nn_sample = bnn()
        nn_sample.train()
        return nn_sample

    return guide


class BNN(BOModel):
    def __init__(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], None],
        guide: Callable[[Optional[torch.Tensor], Optional[torch.Tensor]], nn.Module],
        n_inputs: int,
        n_hidden: int = 100,
        non_linearity: str = "ReLU",
        batch_size: int = 200,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
    ):
        self.model = model
        self.guide = guide
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.non_linearity = non_linearity
        self.batch_size = batch_size
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.optimizer = None
        self.svi = None

    @classmethod
    def get_model(
        cls: Type[_BNN],
        n_inputs: int,
        n_hidden: int = 100,
        non_linearity: str = "ReLU",
        batch_size: int = 200,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        device="cpu",
    ) -> _BNN:
        model = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            getattr(nn, non_linearity)(),
            nn.Linear(n_hidden, 1),
        ).to(device)

        priors = lambda: normal_priors(model, prior_mean, prior_std)
        variational_dists = lambda: normal_variationals(model, prior_mean, prior_std)

        bnn_model = make_bnn_model(model, priors, batch_size=batch_size)
        guide = make_guide(model, variational_dists)
        return cls(
            bnn_model,
            guide,
            n_inputs,
            n_hidden,
            non_linearity,
            batch_size,
            prior_mean,
            prior_std,
        )

    def train_model(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        n_epochs: int,
        optimizer: Optional = None,
        loss: Optional = None,
    ) -> List[float]:
        loss = loss or pyro.infer.Trace_ELBO()
        optimizer = optimizer or self.optimizer or pyro.optim.Adam({})
        self.optimizer = optimizer
        svi = pyro.infer.SVI(self.model, self.guide, optimizer, loss)

        if n_epochs == -1:
            n_steps = 10000000000000
        else:
            n_steps = int(len(inputs) / self.batch_size * n_epochs)

        self.svi = svi  # keep a reference in case of partial training
        return self._train(svi, n_steps, inputs, labels)

    def partial_train_model(self, inputs, labels, n_steps: int, optimizer):
        if self.svi is None:
            svi = pyro.infer.SVI(
                self.model, self.guide, optimizer, loss=pyro.infer.Trace_ELBO()
            )
            self.svi = svi
        self._train(svi, n_steps, inputs, labels)

    @staticmethod
    def _train(
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

    def predict(self, inputs: torch.Tensor, n_samples: int) -> torch.Tensor:
        preds = []

        with torch.no_grad():
            for _ in range(n_samples):
                nn_sample = self.guide()
                nn_sample.eval()
                preds.append(nn_sample(inputs).cpu().squeeze())
        return torch.stack(preds)

    def save_model(self, fname: str, optimizer: Optional = None) -> None:
        """
        :param fname: base path of location to save model. The weights (parameters) will
          be saved at `{fname}.params`; the optimizer parameters, if given, at
          `{fname}.opt`; and the kwargs for `get_model` at `{fname}.pkl`
        """
        pyro.get_param_store().save(f"{fname}.params")

        optimizer = optimizer or self.optimizer

        if optimizer:
            optimizer.save(f"{fname}.opt")

        kwargs = {
            "n_inputs": self.n_inputs,
            "n_hidden": self.n_hidden,
            "non_linearity": self.non_linearity,
            "batch_size": self.batch_size,
            "prior_mean": self.prior_mean,
            "prior_std": self.prior_std,
        }

        with open(f"{fname}.pkl", "wb") as f:
            pickle.dump(kwargs, f)

    @classmethod
    def load_model(
        cls: Type[_BNN],
        fname: str,
        device: str = "cpu",
        optimizer_func: Optional[Callable] = None,
    ) -> Union[_BNN, Tuple[_BNN, Any]]:

        with open(f"{fname}.pkl", "rb") as f:
            kwargs = pickle.load(f)

        model = cls.get_model(**kwargs)

        optimizer = (
            optimizer_func() if optimizer_func else pyro.optim.Adam({"lr": 0.01})
        )

        pyro.get_param_store().load(f"{fname}.params")
        optimizer.load(f"{fname}.opt")

        return model, optimizer
