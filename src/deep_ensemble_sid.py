"""
Deep ensemble model from [this paper]().

Only a simple MLP is supported at present, but it should be easy to ensemble other
`torch.nn.Module`s by using them instead of `NN` in `NNEnsemble`.

This implementation also only supports training all models in the ensemble at once
on the same GPU; modifications would be needed to use this for larger models.
"""

import pickle
import numpy as np
import torch
import copy
import torch.nn as nn
from torch.nn import Linear, ReLU, Softplus,Dropout
from torch.utils.data import TensorDataset, DataLoader
import torch.distributions as tdist
from itertools import cycle
from typing import Tuple, Optional, Dict, Callable, Sequence, Union, Any, Type, TypeVar
from bb_opt.src.non_matplotlib_utils import save_checkpoint, load_checkpoint
from bb_opt.src.networks.wide_resnet_sid import Wide_ResNet
import bayesian_opt as bopt
import sys

_NNEnsemble = TypeVar("NNEnsemble", bound="NNEnsemble")


def sample_uniform(out_size):
    z = np.zeros((8*out_size,4))
    z[range(8*out_size),np.random.randint(4,size=8*out_size)] = 1
    out_data = torch.from_numpy(z).view((-1,32)).float().cuda()
    return out_data

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
        mean =torch.sigmoid(output[:,0])
        variance =torch.sigmoid(output[:,1])*0.1+self.min_variance
        #variance =torch.sigmoid(output[:,1])*0.01+self.min_variance
        #mean = output[:, 0]
        #variance = self.softplus(output[:, 1]) + self.min_variance
        return mean, variance

class NN_bk(torch.nn.Module):
    def __init__(self, 
            n_inputs: int, 
            n_hidden: int, 
            min_variance: float = 1e-5,
            sigmoid_coeff: float = 1.,
            separate_mean_var=False,
            ):
        super().__init__()
        if separate_mean_var:
            self.hidden = [Linear(n_inputs, n_hidden), Linear(n_inputs, n_hidden)]
            self.output = [Linear(n_hidden, 1), Linear(n_hidden, 1)]
            self.hidden = nn.ModuleList(self.hidden)
        else:
            if n_hidden > 0:
                self.hidden = [Linear(n_inputs, n_hidden)]
                self.output = [Linear(n_hidden, 2)]
                self.hidden = nn.ModuleList(self.hidden)
            else:
                self.output = [Linear(n_inputs, 2)]
        self.output = nn.ModuleList(self.output)

        self.non_linearity = ReLU()
        self.softplus = Softplus()
        self.min_variance = min_variance
        self.sigmoid_coeff = sigmoid_coeff
        self.n_hidden = n_hidden
        #self.pred_var_out = nn.Softplus()
        self.pred_var_out = torch.sigmoid

    def forward(self, x):
        hidden = []
        output = []
        for i in range(len(self.output)):
            if self.n_hidden > 0:
                hidden += [self.non_linearity(self.hidden[i](x))]
                output += [self.output[i](hidden[-1])]
            else:
                output += [self.output[i](x)]

        if len(output) == 1:
            if self.sigmoid_coeff > 0:
                mean = torch.sigmoid(output[0][:, 0])*self.sigmoid_coeff
            else:
                mean = output[:, 0]
            variance = self.pred_var_out(output[0][:, 1])*0.1+self.min_variance
        else:
            if self.sigmoid_coeff > 0:
                mean = torch.sigmoid(output[0])*self.sigmoid_coeff
            else:
                mean = output[0]
            variance = self.pred_var_out(output[1])*0.1+self.min_variance

        #variance = self.softplus(output[:, 1]) + self.min_variance
        return mean.view(-1), variance.view(-1)

    def reset_parameters(self):
        for h in self.hidden:
            h.reset_parameters()
        for o in self.output:
            o.reset_parameters()

class NN2(torch.nn.Module):
    def __init__(self, 
			n_inputs: int, 
			n_hidden: int, 
			min_variance: float = 1e-5,
            sigmoid_coeff: float = 1.,
            separate_mean_var=False,
			):
        super().__init__()
        self.hidden1 = Linear(n_inputs, n_hidden)
        self.hidden2 = Linear(n_hidden, n_hidden)
        self.output = Linear(n_hidden, 2)
        self.non_linearity = non_linearity()
        self.dropout = Dropout(0.5)
        self.min_variance = min_variance
        self.sigmoid_coeff = sigmoid_coeff

    def forward(self, x):
        hidden1 = self.non_linearity(self.hidden1(x))
        hidden2 = self.non_linearity(self.hidden2(hidden1))
        dropout2 = self.dropout(hidden2)
        output = self.output(dropout2)
        if self.sigmoid_coeff > 0:
            mean = torch.sigmoid(output[:,0])*self.sigmoid_coeff
        else:
            mean = output[:,0]
        variance =torch.sigmoid(output[:,1])*0.1+self.min_variance
        #mean = output[:, 0]
        #variance = self.softplus(output[:, 1]) + self.min_variance
        return mean, variance

    def reset_parameters(self):
        self.hidden1.reset_parameters()
        self.hidden2.reset_parameters()
        self.output.reset_parameters()


class RandomNN(torch.nn.Module):
    """
    Single-layer MLP that predicts a Gaussian for each point.
    """

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        weight_min: Optional[float] = None,
        weight_max: Optional[float] = None,
        non_linearity: Callable = torch.nn.ReLU,
        min_variance: float = 1e-5,
        c: list=[1.0,0.1]
    ):
        super().__init__()
        self.hidden1 = Linear(n_inputs, n_hidden)
        self.hidden2 = Linear(n_hidden, n_hidden)
        self.output = Linear(n_hidden, 2)
        self.non_linearity = non_linearity()
        self.dropout = Dropout(0.5)
        self.softplus = Softplus()
        self.min_variance = min_variance
        self.c = c

        if weight_max is None and weight_min and weight_min < 0:
            weight_max = -weight_min
        if weight_min is None and weight_max and weight_max > 0:
            weight_min = -weight_max

        if weight_min:
            self.apply(lambda module: uniform_weights(module, weight_min, weight_max))
            self.weight_min = weight_min
            self.weight_max = weight_max

    def forward(self, x):
        hidden1 = self.non_linearity(self.hidden1(x))
        hidden2 = self.non_linearity(self.hidden2(hidden1))
        dropout2 = self.dropout(hidden2)
        output = self.output(dropout2)
        mean =torch.sigmoid(output[:,0])*self.c[0]
        variance =torch.sigmoid(output[:,1])*self.c[1]+self.min_variance
        return mean, variance

    def reset_parameters(self):
        self.hidden1.reset_parameters()
        self.hidden2.reset_parameters()
        self.output.reset_parameters()


def uniform_weights(module, min_val: float = -5.0, max_val: float = 5.0):
    if isinstance(module, Linear):
        module.weight.data.uniform_(min_val, max_val)

        if module.bias is not None:
            module.bias.data.uniform_(min_val, max_val)


class NNEnsemble(torch.nn.Module):

    def __init__(
        self,
        n_models,
        model_generator,
        model_kwargs_generator,
        device=None,
        adversarial_epsilon=None,
        mu_prior=None,
        std_prior=None,
    ):
        super().__init__()

        try:
            model_generator = cycle(model_generator)
        except TypeError:  # not iterable
            model_generator = cycle([model_generator])

        # don't make an iterator over the dict keys
        if isinstance(model_kwargs_generator, dict):
            model_kwargs_generator = cycle([model_kwargs_generator])
        else:
            try:
                model_kwargs_generator = cycle(model_kwargs_generator)
            except TypeError:  # not iterable
                model_kwargs_generator = cycle([model_kwargs_generator])

        self.n_models = n_models
        self.models = torch.nn.ModuleList(
            [
                next(model_generator)(**next(model_kwargs_generator))
                for _ in range(n_models)
            ]
        )
        self.adversarial_epsilon = adversarial_epsilon
        self.mu_prior = mu_prior
        self.std_prior = std_prior
        self.anchor_models = []
        if mu_prior is not None:
            self.anchor_models = [copy.deepcopy(model).to(device) for model in self.models]
            self.generate_new_anchors()

    def generate_new_anchors(self):
        with torch.no_grad():
            for model in self.anchor_models:
                for param in model.parameters():
                    param.normal_(self.mu_prior, self.std_prior)

    def bayesian_ensemble_loss(self, data_noise):
        if len(self.anchor_models) == 0:
            return 0.

        assert data_noise > 0
        l2 = [0.] * len(self.anchor_models)
        for i in range(len(self.anchor_models)):
            normal_model = self.models[i]
            anchor_model = self.anchor_models[i]
            normal_params = [param for param in normal_model.parameters()]
            anchor_params = [param for param in anchor_model.parameters()]

            n_params = len(normal_params)
            assert n_params == len(anchor_params)

            for j in range(n_params):
                l2[i] += data_noise/self.std_prior * torch.sum((normal_params[j]-anchor_params[j])**2)

        return torch.sum(torch.tensor(l2).cuda())

    def reset_parameters(self):
        for model in self.models:
            model.reset_parameters()

    def forward_old(self, x, individual_predictions: bool = True, all_pairs: bool = True):
        if not all_pairs:
            N = x.shape[0]
            m = len(self.models)
            per_model = N//m
            batches = [i*per_model for i in range(m)] + [N]
            assert len(batches) == m+1

            means = []
            variances = []
            for batch_iter in range(len(batches)-1):
                bs = batches[batch_iter]
                be = batches[batch_iter+1]
                assert be-bs > 0, str(be) + "-" + str(bs) + "; " + str(batch_iter) + "; " + str(batches) + "; " + str(N) + "; " + str(m)

                bmeans, bvariances = self.models[batch_iter](x[bs:be])
                means += [bmeans]
                variances += [bvariances]

            means = torch.cat(means, dim=0)
            variances = torch.cat(variances, dim=0)

            return means, variances
    def forward(self, x, y=None, optimizer=None, individual_predictions: bool = True):
        if y is not None and self.adversarial_epsilon is not None:
            x.requires_grad_()
            means, variances = self(x)
            negative_log_likelihood = self.compute_negative_log_likelihood(
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

        return self.combine_means_variances(means, variances)


    def bagging_forward(self, x, num_to_bag=4):
        nn_idx = np.random.choice(len(self.models), size=4, replace=False)
        models = [self.models[i] for i in nn_idx]

        means, variances = list(zip(*[models[i](x) for i in range(len(models))]))
        means, variances = torch.stack(means), torch.stack(variances)

        return means, variances


    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pred_means, pred_vars = self(inputs)

        pred_means = pred_means.cpu()
        return pred_means

    @staticmethod
    def combine_means_variances(
        means: torch.Tensor, variances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = means.mean(dim=0)
        variance = (variances + means ** 2).mean(dim=0) - mean ** 2
        return mean, variance

    @staticmethod
    def compute_negative_log_likelihood(
        labels, 
        means, 
        variances, 
        custom_std=None,
        return_mse=False,
    ):
        if custom_std is not None:
            variances = torch.tensor(custom_std**2).cuda()
        mse = (labels - means) ** 2
        negative_log_likelihood = 0.5 * (torch.log(variances) + mse / variances)
        negative_log_likelihood = negative_log_likelihood.mean(dim=-1).sum()
        print(negative_log_likelihood,file=sys.stderr) 
        if return_mse:
            mse = mse.mean(dim=-1).mean()
            return negative_log_likelihood, mse
        return negative_log_likelihood

    @staticmethod
    def get_per_point_nll(
        labels, means, variances, return_mse=False
    ):
        mse = (labels - means) ** 2
        negative_log_likelihood = 0.5 * (torch.log(variances) + mse / variances)
        negative_log_likelihood = negative_log_likelihood.mean(dim=0)

        if return_mse:
            mse = mse.mean(dim=-1).sum()
            return negative_log_likelihood, mse
        return negative_log_likelihood

    def report_metric(
        labels, means, variances, custom_std=None, return_mse=False
    ):
        num_samples = means.shape[0]

        d = tdist.Normal(means.view(-1), torch.sqrt(variances.view(-1)))
        negative_log_likelihood1 = -d.log_prob(labels.unsqueeze(0).expand(num_samples, -1).contiguous().view(-1)).mean().detach()

        m, v = bopt.combine_means_variances(means, variances)
        mse_m = ((labels - m) ** 2).detach()
        #if custom_std is None:
        #    d = tdist.Normal(m, torch.sqrt(v))
        #else:
        #    d = tdist.Normal(m, custom_std)
        #negative_log_likelihood2 = -d.log_prob(labels).mean().detach()
        negative_log_likelihood2 = bopt.get_nll(means, torch.sqrt(variances), labels, tdist.Normal, single_gaussian=True).mean().detach()

        if return_mse:
            return negative_log_likelihood1, negative_log_likelihood2, mse_m.mean()
        return negative_log_likelihood1, negative_log_likelihood2

    def report_metric_sb(
        labels, means, variances, custom_std=None, return_mse: bool = False
    ):  
        if custom_std is not None:
            variances = torch.tensor(custom_std**2).cuda()
        m = means.mean(dim=0)
        v = (variances + means ** 2).mean(dim=0)-m ** 2
        mse_m= (labels - m) ** 2
        mse = (labels - means) ** 2
        negative_log_likelihood1 = 0.5 * (torch.log(variances) + mse / variances)
        negative_log_likelihood1 = negative_log_likelihood1.mean(dim=-1).mean()
        negative_log_likelihood2 = 0.5*(torch.log(v)+ mse_m/v)
        negative_log_likelihood2 = negative_log_likelihood2.mean()

        if return_mse:
            return negative_log_likelihood1,negative_log_likelihood2, mse_m.mean()
        return negative_log_likelihood1,negative_log_likelihood2

    @staticmethod
    def compute_weighted_nll(labels, means, variances, return_mse: bool = False):
        mse = (labels - means) ** 2
        negative_log_likelihood = 0.5 * (torch.log(variances) + mse / variances)
        negative_log_likelihood = negative_log_likelihood.mean(dim=-1)
        negative_log_likelihood = (labels * negative_log_likelihood).sum()

        if return_mse:
            mse = mse.mean(dim=-1).sum()
            return negative_log_likelihood, mse
        return negative_log_likelihood

    @classmethod
    def get_model(
        cls,
        n_inputs: int,
        batch_size: int = 200,
        n_models: int = 5,
        n_hidden: int = 100,
        adversarial_epsilon: Optional = None,
        device=None,
        nonlinearity_names: Sequence[str] = None,
        extra_random: bool = False,
        single_layer:bool = True,
        sigmoid_coeff : float = 1.,
        separate_mean_var = False,
        mu_prior=None,
        std_prior=None,
    ):
        device = device or "cpu"
        assert not extra_random, "Not implemented"
        if not extra_random:
            model_kwargs = {
                    "n_inputs": n_inputs, 
                    "n_hidden": n_hidden, 
                    #"sigmoid_coeff": sigmoid_coeff, 
                    #"separate_mean_var": separate_mean_var,
                   }
            if single_layer:
                model = cls(
                    n_models, 
                    NN, 
                    model_kwargs, 
                    device=device,
                    adversarial_epsilon=adversarial_epsilon,
                    mu_prior=mu_prior,
                    std_prior=std_prior,
                ).to(device)
            else:
                model = cls(
                    n_models, 
                    NN2, 
                    model_kwargs, 
                    device=device,
                    adversarial_epsilon=adversarial_epsilon,
                    mu_prior=mu_prior,
                    std_prior=std_prior,
                ).to(device)
            return model

        def gelu(x):
            return 0.5 * x * ( 1 + torch.tanh( torch.sqrt(2 / x.new_tensor(np.pi)) * (x + 0.044715 * x ** 3)))

        def swish(x):
            return x * torch.sigmoid(x)

        non_linearities = {
            "ELU": torch.nn.ELU,
            #     torch.nn.GLU,
            #     torch.nn.Hardshrink,
            "Hardtanh": torch.nn.Hardtanh,
            "LeakyReLU": torch.nn.LeakyReLU,
            "PReLU": torch.nn.PReLU,
            "ReLU": torch.nn.ReLU,
            "SELU": torch.nn.SELU,
            "Sigmoid": torch.nn.Sigmoid,
            "Softmin": torch.nn.Softmin,  # /
            "Softplus": torch.nn.Softplus,
            #     torch.nn.Softshrink,
            "Softsign": torch.nn.Softsign,
            "Tanh": torch.nn.Tanh,
            "Tanhshrink": torch.nn.Tanhshrink,
            "gelu": lambda: gelu,
            "swish": lambda: swish,
        }

        if nonlinearity_names:
            assert len(nonlinearity_names) == n_models
            random_nonlinearity = False
            non_linearities = [non_linearities[name] for name in nonlinearity_names]
        else:
            random_nonlinearity = True
            non_linearities = list(non_linearities.values())

        def model_kwargs():
            kwargs = {"n_inputs": n_inputs, "n_hidden": n_hidden}
            non_linearity_idx = 0
            while True:
                kw = kwargs.copy()
                kw["weight_max"] = np.random.uniform(0.1, 20)

                if random_nonlinearity:
                    kw["non_linearity"] = np.random.choice(non_linearities)
                else:
                    kw["non_linearity"] = non_linearities[non_linearity_idx]
                    non_linearity_idx += 1

                yield kw

        model = cls(
            n_models, RandomNN, model_kwargs(), adversarial_epsilon=adversarial_epsilon
        ).to(device)
        return model

    def train_model(
        self,
        inputs,
        labels,
        n_epochs,
        batch_size,
        optimizer_kwargs: Optional[Dict] = None,
    ):
        optimizer_kwargs = optimizer_kwargs or {}
        data = TensorDataset(inputs, labels)
        loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), **optimizer_kwargs)

        self.train()
        for epoch in range(n_epochs):
            for batch in loader:
                inputs, labels = batch
                optimizer.zero_grad()

                means, variances = self(inputs)

                negative_log_likelihood = self.compute_negative_log_likelihood(
                    labels, means, variances
                )
                negative_log_likelihood.backward()
                optimizer.step()
        self.eval()

    def train_model_var(
        self,
        inputs,
        labels,
        n_epochs,
        batch_size,
    out_size,
    out_weight,
        optimizer_kwargs: Optional[Dict] = None,
    ):
        optimizer_kwargs = optimizer_kwargs or {}
        data = TensorDataset(inputs, labels)
        loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), **optimizer_kwargs)
        self.train()
        for epoch in range(n_epochs):
            for batch in loader:
                inputs, labels = batch
                optimizer.zero_grad()
                means, variances = self(inputs)

                negative_log_likelihood = self.compute_negative_log_likelihood(
                    labels, means, variances
                )
                z = np.zeros((8*out_size,4))
                z[range(8*out_size),np.random.randint(4,size=8*out_size)]=1
                out_data = torch.from_numpy(z).view((-1,32))
                means_o, variances_o = self(out_data)
                loss=negative_log_likelihood+out_weight*(means_o.var(dim=0).mean())
                loss.backward()
                optimizer.step()
        self.eval()

    def train_model_mean(
        self,
        inputs,
        labels,
        n_epochs,
        batch_size,
        out_size,
        out_weight,
		default_mean,
        optimizer_kwargs: Optional[Dict] = None,
    ):
        optimizer_kwargs = optimizer_kwargs or {}
        data = TensorDataset(inputs, labels)
        loader = DataLoader(data, batch_size=batch_size, shuffle=True)  
        optimizer = torch.optim.Adam(self.parameters(), **optimizer_kwargs)
        self.train()
        for epoch in range(n_epochs):
            for batch in loader:
                inputs, labels = batch
                optimizer.zero_grad()  
                means, variances = self(inputs)

                negative_log_likelihood = self.compute_negative_log_likelihood(
                    labels, means, variances
                )
                z = np.zeros((8*out_size,4))
                z[range(8*out_size),np.random.randint(4,size=8*out_size)]=1
                out_data = torch.from_numpy(z).view((-1,32))
                means_o, variances_o = self(out_data)
                loss=negative_log_likelihood+out_weight*(self.compute_negative_log_likelihood(default_mean,means_o,variances_o))
                loss.backward()
                optimizer.step()
        self.eval()

    def save_model(self, fname: str, optimizer: Optional = None) -> None:
        """
        WARNING - saving/loading an ensemble using this function assumes that each model
        in the ensemble has the same number of hidden units and that the ensemble is
        constructable by `NNEnsemble.get_model`.

        :param fname: path to .pth file to which to save weights
        A .pkl file with the same base name/path will be used to save the
        nonlinearity names and a few other variables.
        """
        nonlinearity_names = []
        for m in self.models:
            try:
                name = m.non_linearity.__name__
            except AttributeError:
                name = type(m.non_linearity).__name__
            nonlinearity_names.append(name)

        kwargs = {
            # all have to take the same input shape
            "n_inputs": next(m.children()).in_features,
            "n_models": self.n_models,
            # assumption: all have same hidden size; true in the models I use (so far)
            "n_hidden": next(m.children()).out_features,
            "adversarial_epsilon": self.adversarial_epsilon,
            "nonlinearity_names": nonlinearity_names,
        }

        with open(fname.replace(".pth", ".pkl"), "wb") as f:
            pickle.dump(kwargs, f)

        save_checkpoint(fname, self, optimizer)

    @classmethod
    def get_model_resnet(
        cls,
        n_inputs: int,
        n_models: int = 5,
        depth: int = 16,
        widen_factor: int=8,
        dropout: float=0.3,
        device=None,
        extra_random: bool = False,
        mu_prior=None,
        std_prior=None,
    ):
        device = device or "cpu"

        model_kwargs = {
                "depth": depth,
                "widen_factor": widen_factor,
                "dropout_rate":dropout,
                } # args.depth, args.widen_factor, args.dropout, num_classes
        model = cls(   
                    n_models, 
                    Wide_ResNet, 
                    model_kwargs, 
                    device=device,
                    adversarial_epsilon=None,
                    mu_prior=mu_prior,
                    std_prior=std_prior,
                ).to(device)
        return model

    @classmethod
    def load_model(
        cls: Type[_NNEnsemble],
        fname: str,
        device: str = "cpu",
        optimizer_func: Optional[Callable] = None,
    ) -> Union[_NNEnsemble, Tuple[_NNEnsemble, Any]]:
        """
        WARNING - saving/loading an ensemble using this function assumes that each model
        in the ensemble has the same number of hidden units and that the ensemble is
        constructable by `NNEnsemble.get_model`.

        :param fname: path to .pth file with weights to load
        There must also be a .pkl file with the same base name/path with
        a list of the activation function names to use.
        :param device: device onto which to load the model
        :optimizer_func: a function which takes in model parameters and returns an optimizer
        If None, Adam is used (with lr=0.01).
        :returns: (model, optimizer) if optimizer state was saved otherwise model
        """
        batch_size = 1  # this isn't used

        with open(fname.replace(".pth", ".pkl"), "rb") as f:
            kwargs = pickle.load(f)
            n_models = kwargs["n_models"]
            n_inputs = kwargs["n_inputs"]
            n_hidden = kwargs["n_hidden"]
            adversarial_epsilon = kwargs["adversarial_epsilon"]
            nonlinearity_names = kwargs["nonlinearity_names"]

        model = cls.get_model(
            n_inputs,
            batch_size,
            n_models,
            n_hidden,
            adversarial_epsilon,
            device,
            nonlinearity_names,
        )

        try:
            optimizer = (
                optimizer_func(model.parameters())
                if optimizer_func
                else torch.optim.Adam(model.parameters(), lr=0.01)
            )
            load_checkpoint(fname, model, optimizer)
            return model, optimizer
        except KeyError:
            load_checkpoint(fname, model)
            return model


class ResnetEnsemble(torch.nn.Module):
    def __init__(self,
        params,
        n_models,
        n_inputs,
        depth=16,
        widen_factor=8,
        n_hidden=100,
        dropout_factor=0.0,
        mu_prior=None,
        std_prior=None,
    ):
        super().__init__()
        assert params.ensemble_type == "fc", params.ensemble_type

        self.conv_model = Wide_ResNet(depth, widen_factor, dropout_factor, fc_sampling=True)
        self.fc_layers = NNEnsemble.get_model(
                self.conv_model.nStages[3],
                n_models,
                n_hidden,
                device=params.device,
                extra_random=False,
                )

        print('num resnet params:', sum(p.numel() for p in self.conv_model.parameters()))
        print('num fc ensemble params:', sum(p.numel() for p in self.fc_layers.parameters()))

        self.mu_prior = mu_prior
        self.std_prior = std_prior
        self.anchor_models = []
        if mu_prior is not None:
            self.anchor_models = [copy.deepcopy(model) for model in self.fc_layers.models] 
            self.generate_new_anchors()

    def generate_new_anchors(self):
        for model in self.anchor_models:
            for param in model.parameters():
                param.normal_(self.mu_prior, self.std_prior)

    def bayesian_ensemble_loss(self, data_noise):
        if len(self.anchor_models) == 0:
            return 0.

        l2 = [0.] * len(self.anchor_models)
        for i in range(len(self.anchor_models)):
            normal_model = self.fc_layers.models[i]
            anchor_model = self.anchor_models[i]
            normal_params = [param for param in normal_model.parameters()]
            anchor_params = [param for param in anchor_model.parameters()]

            n_params = len(normal_params)
            assert n_params == len(anchor_params)

            for j in range(n_params):
                l2[i] += data_noise/self.std_prior * torch.sum((normal_params[j]-anchor_params[j])**2)

        return torch.sum(l2)

    def fc_input_size(self):
        return self.conv_model.nStages[3]

    def freeze_conv(self):
        for param in self.conv_model.parameters():
            param.requires_grad = False

    def unfreeze_conv(self):
        for param in self.conv_model.parameters():
            param.requires_grad = True

    def forward(self, x):
        fc_input = self.conv_model(x)
        out = self.fc_layers.forward(fc_input)
        return out

    def conv_forward(self, x, batch_size=0):
        if batch_size > 0:
            N = x.shape[0]
            num_batches = N//batch_size+1
            batches = [i*batch_size  for i in range(num_batches)] + [N]
            out = []
            for bi in range(num_batches):
                bs = batches[bi]
                be = batches[bi+1]
                bN = be-bs
                if bN <= 0:
                    continue
                out += [self.conv_model(x[bs:be])]
            out = torch.cat(out, dim=0)
            return out
        else:
            return self.conv_model(x)

    def fc_forward(self, fc_input):
        out = self.fc_layers.forward(fc_input)
        return out

    def reset_parameters(self):
        self.conv_model.reset_parameters()
        self.fc_layers.reset_parameters()

    @staticmethod
    def combine_means_variances(means, variances):
        return NNEnsemble.combine_means_variances(means, variances)

    @staticmethod
    def compute_negative_log_likelihood(
        labels, 
        means, 
        variances, 
        custom_std=None,
        return_mse=False,
    ):
        return NNEnsemble.compute_negative_log_likelihood(
                labels,
                means,
                variances,
                custom_std,
                return_mse
                )

    @staticmethod
    def report_metric(labels, means, variances, custom_std=None, return_mse=False):
        return NNEnsemble.fc_layers(
                labels,
                means,
                variances,
                custom_std,
                return_mse
                )


class ResnetEnsemble2(torch.nn.Module):
    def __init__(self,
        params,
        n_models,
        n_inputs,
        depth=16,
        widen_factor=8,
        n_hidden=100,
        dropout_factor=0.0,
        device='cuda',
        mu_prior=None,
        std_prior=None,
    ):
        super().__init__()
        assert params.ensemble_type == "fc", params.ensemble_type

        self.conv_model = nn.ModuleList([Wide_ResNet(depth, widen_factor, dropout_factor, fc_sampling=True, do_batch_norm=params.resnet_do_batch_norm) for i in range(n_models)])
        self.fc_layers = nn.ModuleList([NN(self.conv_model[0].nStages[3], n_hidden) for i in range(n_models)])

        print('num resnet params:', sum(p.numel() for p in self.conv_model[0].parameters()) * n_models)
        print('num fc ensemble params:', sum(p.numel() for p in self.fc_layers[0].parameters()) * n_models)

        self.mu_prior = mu_prior
        self.std_prior = std_prior
        self.anchor_models = []
        if mu_prior is not None:
            self.anchor_models = [[copy.deepcopy(self.conv_model[i]).to(device), copy.deepcopy(self.fc_layers[i]).to(device)] for i in range(n_models)] 
            self.generate_new_anchors()

    def generate_new_anchors(self):
        with torch.no_grad():
            for model_pair in self.anchor_models:
                for model in model_pair:
                    for param in model.parameters():
                        param.normal_(self.mu_prior, self.std_prior)

    def bayesian_ensemble_loss(self, data_noise):
        if len(self.anchor_models) == 0:
            return 0.

        l2 = [0.] * len(self.anchor_models)
        for i in range(len(self.anchor_models)):
            normal_model = [self.conv_model[i], self.fc_layers[i]]
            anchor_model = self.anchor_models[i]

            for k in range(2):
                normal_params = [param for param in normal_model[k].parameters()]
                anchor_params = [param for param in anchor_model[k].parameters()]

                n_params = len(normal_params)
                assert n_params == len(anchor_params)

                for j in range(n_params):
                    l2[i] += data_noise/self.std_prior * torch.sum((normal_params[j]-anchor_params[j])**2)
        l2 = torch.tensor(l2)

        return torch.sum(l2)

    def fc_input_size(self):
        return self.conv_model[0].nStages[3]

    def freeze_conv(self):
        for i in range(len(self.conv_model)):
            self.conv_model[i].eval()
            for param in self.conv_model[i].parameters():
                param.requires_grad = False

    def unfreeze_conv(self):
        for i in range(len(self.conv_model)):
            self.conv_model[i].train()
            for param in self.conv_model[i].parameters():
                param.requires_grad = True


    def freeze_fc(self):
        for i in range(len(self.fc_layers)):
            self.fc_layers[i].eval()
            for param in self.fc_layers[i].parameters():
                param.requires_grad = False

    def unfreeze_fc(self):
        for i in range(len(self.fc_layers)):
            self.fc_layers[i].eval()
            for param in self.fc_layers[i].parameters():
                param.requires_grad = False

    def freeze(self):
        self.freeze_conv()
        self.freeze_fc()

    def unfreeze(self):
        self.unfreeze_conv()
        self.unfreeze_fc()

    def forward(self, x, return_fc_input=False):
        n_models = len(self.conv_model)
        fc_input = [self.conv_model[i](x) for i in range(n_models)]
        out_mean = []
        out_var = []
        for i in range(n_models):
            out = self.fc_layers[i](fc_input[i])
            out_mean += [out[0]]
            out_var += [out[1]]

        if return_fc_input:
            return torch.stack(out_mean, dim=0), torch.stack(out_var, dim=0), torch.stack([t.detach() for t in fc_input])
        else:
            return torch.stack(out_mean, dim=0), torch.stack(out_var, dim=0)

    def conv_forward(self, x, batch_size=0):
        n_models = len(self.conv_model)
        if batch_size > 0:
            N = x.shape[0]
            num_batches = N//batch_size+1
            batches = [i*batch_size  for i in range(num_batches)] + [N]
            out = [[] for i in range(n_models)]
            for bi in range(num_batches):
                bs = batches[bi]
                be = batches[bi+1]
                bN = be-bs
                if bN <= 0:
                    continue
                for i in range(n_models):
                    out[i] += [self.conv_model[i](x[bs:be])]
            for i in range(n_models):
                out[i] = torch.cat(out, dim=0)
            return torch.stack(out, dim=0)
        else:
            return torch.stack([self.conv_model[i](x) for i in range(n_models)], dim=0)

    def fc_forward(self, fc_input):
        n_models = len(self.conv_model)
        assert fc_input.shape[0] == len(self.conv_model)
        out_mean = []
        out_var = []
        for i in range(n_models):
            out = self.fc_layers[i](fc_input[i])
            out_mean += [out[0]]
            out_var += [out[1]]
        return torch.stack(out_mean, dim=0), torch.stack(out_var, dim=0)

    def reset_parameters(self):
        for i in range(len(self.conv_model)):
            self.conv_model[i].reset_parameters()
            self.fc_layers[i].reset_parameters()

    @staticmethod
    def combine_means_variances(means, variances):
        return NNEnsemble.combine_means_variances(means, variances)

    @staticmethod
    def compute_negative_log_likelihood(
        labels, 
        means, 
        variances, 
        custom_std=None,
        return_mse=False,
    ):
        return NNEnsemble.compute_negative_log_likelihood(
                labels,
                means,
                variances,
                custom_std,
                return_mse,
                )

    @staticmethod
    def report_metric(labels, means, variances, custom_std=None, return_mse=False):
        return NNEnsemble.fc_layers(
                labels,
                means,
                variances,
                custom_std,
                return_mse
                )
