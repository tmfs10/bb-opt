```python
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pyro
import pyro.optim
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from typing import Union, Tuple, Optional, Callable, Dict, Set, List
from tqdm import tnrange
from bb_opt.src.bayesian_opt import (
    normal_priors, normal_variationals,
    spike_slab_priors, SpikeSlabNormal,
    make_bnn_model, make_guide,
    train, bnn_predict, optimize,
    get_model_bnn, acquire_batch_bnn_greedy, train_model_bnn,
    get_model_nn, acquire_batch_nn_greedy, train_model_nn
)
from bb_opt.src.utils import plot_performance

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
%matplotlib inline

batch_size = 200

ec50 = pd.read_csv('../data/malaria.csv').ec50.values
fingerprints = np.load('../data/fingerprints.npy')
fingerprints.shape
```

```python
train_inputs, test_inputs, train_labels, test_labels = train_test_split(fingerprints, ec50, test_size=.1)

train_inputs = torch.tensor(train_inputs).float().to(device)
train_labels = torch.tensor(train_labels).float().to(device)
train_data = TensorDataset(train_inputs, train_labels)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_inputs = torch.tensor(test_inputs).float().to(device)
test_labels = torch.tensor(test_labels).float().to(device)
```

```python
n_hidden = 100
non_linearity = 'ReLU'
n_inputs = train_inputs.shape[1]

model = nn.Sequential(
    nn.Linear(n_inputs, n_hidden),
    getattr(nn, non_linearity)(),
    nn.Linear(n_hidden, 1)
).to(device)
```

# Predictive Model

```python
optimizer = torch.optim.Adam(model.parameters())
loss_func = nn.MSELoss()
```

```python
for epoch in range(15):
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()

        predictions = model(inputs).squeeze()
        loss = loss_func(predictions, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(test_inputs).squeeze()
        test_loss = loss_func(predictions, test_labels)

    if epoch % 5 == 0:
        print(f'[E{epoch}] Loss = {loss.item():.3f}. Val loss = {test_loss.item():.3f}')
```

```python
sns.jointplot(model(train_inputs).detach().cpu().squeeze().numpy(), train_labels.cpu().numpy())
```

```python
sns.jointplot(predictions.cpu().numpy(), test_labels.cpu().numpy())
```

```python
# .99 .4
```

## Bayesian

```python
std1 = 0.03
std2 = 1.5
alpha = 0.7

param = torch.tensor(0.)

n1 = pyro.distributions.Normal(0, std1)
n2 = pyro.distributions.Normal(0, std2)
s = SpikeSlabNormal(param, std1, std2, alpha)
x = torch.linspace(-4, 4, 1000)

plt.plot(x.numpy(), n1.log_prob(x).exp().numpy(), label='n1')
plt.plot(x.numpy(), n2.log_prob(x).exp().numpy(), label='n2')
plt.plot(x.numpy(), s.log_prob(x).exp().numpy(), label='spike slab')
plt.plot(x.numpy(),
         (alpha * n1.log_prob(x).exp() + (1 - alpha) * n2.log_prob(x).exp()).numpy(),
         label='mix', ls='dashed')
plt.legend()
plt.ylim(0, 0.5);
```

```python
n_samples = 10
prior_mean = 0
prior_std = .05

priors = lambda: normal_priors(model, prior_mean, prior_std)
# priors = lambda: spike_slab_priors(model, std2=1.5)
variationals = lambda: normal_variationals(model, prior_mean, prior_std)
bnn_model = make_bnn_model(model, priors, batch_size=batch_size)
guide = make_guide(model, variationals)
# guide = AutoDiagonalNormal(model)
```

```python
from bb_opt.src import hsic
```

```python
preds = bnn_predict(guide, test_inputs, n_samples=100)

x = torch.tensor(preds)
kernels = hsic.dimwise_mixrq_kernels(x)
kernels = kernels.double()
```

```python
kernels.shape, x.shape
```

```python
kernels[0, 0]
```

```python
kernels.prod(dim=2)
```

```python
n_samples = [10 * i for i in range(1, 11)]
total_hsics = []

for n_sample in n_samples:
    preds = bnn_predict(guide, test_inputs, n_samples=n_sample)

    x = torch.tensor(preds.T)
    kernels = hsic.dimwise_mixrq_kernels(x)
    kernels = kernels.double()

    n, n_, d = kernels.shape
    assert n == n_, f"First two dimensions must be equal but were {n} and {n_}"

    sum_b = kernels.sum(dim=1)
    t1 = kernels.prod(dim=2).mean()
    t2 = - 2 / (n ** (d + 1)) * torch.sum(torch.prod(sum_b, dim=1))
    sum_ab = torch.sum(sum_b, dim=0)
    t3 = torch.exp(-2 * d * np.log(n) + torch.sum(torch.log(sum_ab)))

    total_hsics.append((t1 + t2 + t3).item())
```

```python
total_hsics = [t.item() for t in total_hsics]
```

```python
plt.plot(n_samples, total_hsics)
plt.yscale('log')
```

```python
th = [t / (10 ** (8 ** np.log10(n_samples[i]))) for i, t in enumerate(total_hsics)]
plt.plot(n_samples, th)
```

```python
from pyro import poutine
```

```python
poutine.block()
poutine.condition()
```

```python
from pyro.infer.mcmc import MCMC, NUTS
```

```python
kernel = NUTS(bnn_model, adapt_step_size=True)
mcmc = MCMC(kernel, num_samples=1000, warmup_steps=200)
```

```python
import inspect
print(inspect.getsource(bnn_model))
```

```python
posterior = mcmc.run(train_inputs, train_labels)
```

```python
optimizer = pyro.optim.Adam({'lr': 0.01})
pyro.clear_param_store()
svi = pyro.infer.SVI(bnn_model, guide, optimizer, pyro.infer.Trace_ELBO(n_samples))

losses = []
```

```python
losses += train(svi, 10_000, train_inputs, train_labels, verbose=True)
```

```python
plt.figure(figsize=(15, 4))

plt.subplot(121)
plt.plot(losses)
plt.subplot(122)
plt.plot(pd.Series(losses[-3000:]).rolling(window=10).median());
```

```python
preds = bnn_predict(guide, train_inputs, n_samples=50)
sns.jointplot(preds.mean(axis=0), train_labels.cpu().numpy())
```

```python
preds = bnn_predict(guide, test_inputs, n_samples=50)
sns.jointplot(preds.mean(axis=0), test_labels.cpu().numpy())
```

```python
# .83 .44 (~40K)
# .98 .27 (~30K) w/ spike-slab not converged yet - this one overfits?
# .95 .39 (30K) w/ spike-slab not converged yet - it varies a bit more too?
# .98 .34 (70K) spike-slab converged
```

# BO

```python
fraction_best = {}  # {model_name: (mean, std)}
n_repeats = 30
top_k_percent = 1
batch_size = 200
```

## Greedy NN

```python
n_epochs = 0
mean, std = optimize(get_model_nn, acquire_batch_nn_greedy, train_model_nn, fingerprints,
                     ec50, top_k_percent, n_repeats, batch_size, n_epochs, device)
fraction_best['Greedy NN (Untrained)'] = (mean, std)

n_epochs = 3
mean, std = optimize(get_model_nn, acquire_batch_nn_greedy, train_model_nn, fingerprints,
                     ec50, top_k_percent, n_repeats, batch_size, n_epochs, device)
fraction_best['Greedy NN'] = (mean, std)
```

```python
plot_performance(fraction_best)
```

## Greedy BNN

```python
n_repeats = 1
n_epochs = 10

mean, std = optimize(get_model_bnn, acquire_batch_bnn_greedy, train_model_bnn, fingerprints,
                     ec50, top_k_percent, n_repeats, batch_size, n_epochs, device)
fraction_best['Greedy BNN'] = (mean, std)
```

```python
import pickle
with open('fraction_best.pkl', 'wb') as f:
    pickle.dump(fraction_best, f)
```

```python
# 200 epochs
plot_performance(fraction_best)
```

```python
# 50 epochs
plot_performance(fraction_best)
```

```python
# 20 epochs
# plot_performance(fraction_best)
```

```python
# 5 epochs
# plot_performance(fraction_best)
```

```python
with open('fraction_best.pkl', 'rb') as f:
    fraction_best = pickle.load(f)
```

```python
import os
import re
```

```python
fnames = [fname for fname in os.listdir('.') if re.fullmatch('fraction_best_n_epochs_\d+_\d+.pkl', fname)]
dicts = []
for fname in fnames:
    with open(fname, 'rb') as f:
        dicts.append(pickle.load(f))

fraction_best_ = {}
for dict_ in dicts:
    for key, (mean, std) in dict_.items():
        if key in fraction_best_:
            fraction_best_[key] += [mean]
        else:
            fraction_best_[key] = [mean]

for key, mean in fraction_best_.items():
    fraction_best_[key] = np.mean(fraction_best_[key], axis=0), np.std(fraction_best_[key], axis=0)

fraction_best.update(fraction_best_)
```

```python
plot_performance(fraction_best)
```

```python
fraction_best['Greedy BNN'][0][[5, 10, 15, 20, 25]]
```

5 -> .2
10 -> .4
15 -> .5
20 -> .55
25 -> .6

## PDTS

```python
mean, std = optimize(get_model_bnn, acquire_batch_pdts, train_model_bnn,
                     fingerprints, data.ec50, top_k_percent, 2, batch_size, 2, device)
fraction_best['PDTS'] = (mean, std)
```

```python
plot_performance(fraction_best)
```

```python
import os
import re
```

```python
data_dir = 'plot_data'
fnames = [fname for fname in os.listdir(data_dir) if re.fullmatch('fraction_best_n_epochs_800_\d+.pkl', fname)]
dicts = []
for fname in fnames:
    with open(os.path.join(data_dir, fname), 'rb') as f:
        dicts.append(pickle.load(f))

fraction_best_ = {}
for dict_ in dicts:
    for key, (mean, std) in dict_.items():
        if key in fraction_best_:
            fraction_best_[key] += [mean]
        else:
            fraction_best_[key] = [mean]

for key, mean in fraction_best_.items():
    fraction_best_[key] = np.mean(fraction_best_[key], axis=0), np.std(fraction_best_[key], axis=0)

fraction_best.update(fraction_best_)
```

```python
fraction_best['PDTS'][0][10], fraction_best['PDTS'][0][20]
```

```python
plot_performance(fraction_best)
```

```python
plot_performance(fraction_best)
```

```python
plot_performance(fraction_best)
```

# Uniform Sampling Baseline

```python
get_model = lambda: lambda inputs: np.random.random(size=len(inputs))
train_model = lambda *args: None
mean, std = greedy_bayesian_opt(get_model, greedy_acquire, train_model, fingerprints, data.ec50,
                                top_k_percent, n_repeats, device)
fraction_best['Uniformly Random'] = (mean, std)
```

```python
plot_performance(fraction_best)
```

```python
top_k_percent = 1
n_repeats = 100

all_fraction_best_sampled = []
for _ in range(n_repeats):
    sampled_molecules = []
    remaining_molecules = list(range(len(data)))
    np.random.shuffle(remaining_molecules)

    n_top_k_percent = int(top_k_percent / 100 * len(data))
    best_molecules = set(data.ec50.sort_values(ascending=False)[:n_top_k_percent].index)
    fraction_best_sampled = []

    while remaining_molecules:
        sampled_molecules += remaining_molecules[:batch_size]
        remaining_molecules = remaining_molecules[batch_size:]
        fraction_best_sampled.append(len(best_molecules.intersection(sampled_molecules)) / len(best_molecules))

    all_fraction_best_sampled.append(fraction_best_sampled)
```

```python
fraction_best_sampled = np.array(all_fraction_best_sampled)
mean_fraction_best = fraction_best_sampled.mean(axis=0)
std_fraction_best = fraction_best_sampled.std(axis=0)
```

```python
plt.plot(mean_fraction_best, label='Random')
plt.fill_between(range(len(mean_fraction_best)),
                 mean_fraction_best - std_fraction_best,
                 mean_fraction_best + std_fraction_best,
                 alpha=0.5
                )
plt.xlabel('# Batches')
plt.ylabel(f'Fraction of top {top_k_percent}% discovered')
plt.title('Malaria Dataset')
plt.legend()
```

# PBP

```python
from probabilistic_backprop import PBP_net
import numpy as np
```

```python
np.random.seed(1)
data = np.loadtxt("/cluster/nhunt/github/probabilistic_backprop/data/boston_housing.txt")
x = data[:, :-1]
y = data[:, -1]
```

```python
# train/test split
idx = np.random.choice(range(len(X)), len(X), replace=False)
n_train = int(len(X) * 0.9)
idx_train = idx[:n_train]
idx_test = idx[n_train:]

x_train = x[idx_train]
y_train = y[idx_train]
x_test = x[idx_test]
y_test = y[idx_test]

# We construct the network with with two-hidden layers
# with 50 neurons in each one and normalizing the training features to have
# zero mean and unit variance in the training set.

n_hidden_units = 50
net = PBP_net(
    x_train, y_train, [n_hidden_units, n_hidden_units], normalize=True, n_epochs=40
)

# We make predictions for the test set

m, v, v_noise = net.predict(X_test)

# We compute the test RMSE

rmse = np.sqrt(np.mean((y_test - m) ** 2))

print(rmse)

# We compute the test log-likelihood

test_ll = np.mean(
    -0.5 * np.log(2 * math.pi * (v + v_noise)) - 0.5 * (y_test - m) ** 2 / (v + v_noise)
)

print(test_ll)
```
