```python
%load_ext autoreload
%autoreload 2
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pyro
import pyro.optim
import pyro.distributions as dist
from scipy.stats import pearsonr
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from antibody_design.src.utils import load_seqs, encode, make_splits
from bayesian_opt import (random_points, optimize_inputs,
                          make_bnn_model, make_guide,
                          normal_priors, normal_variationals,
                          plot_contours
                         )

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pyro.enable_validation()  # slower but better for debugging
%matplotlib inline
```

```python
class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1)

nn.Flatten = Flatten
```

# Ab Model

```python
tasks = ['J3/J2']
mask_val = -999

seqs = load_seqs(tasks, mask_val)
seqs.head()
```

```python
splits = make_splits(seqs.index, seqs[tasks], test_frac=.1, val_frac=.1, by_value=False)
max_seq_len = splits.train.inputs.str.len().max()

for split in splits:
    if len(splits[split].inputs):
        splits[split].inputs = encode(splits[split].inputs.str.pad(max_seq_len, side='right', fillchar='J'))
#         splits[split].inputs = get_prot_vecs(splits[split].inputs, max_seq_len)

splits.train.inputs.shape
```

```python
_, n_features, n_timesteps = splits.train.inputs.shape
cnn_kernels = 64
pool_stride = 2
dense_size = 32

model = nn.Sequential(
    nn.Conv1d(n_features, cnn_kernels, 5, padding=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(cnn_kernels * n_timesteps, dense_size),
    nn.ReLU(),
#     nn.Dropout(),
    nn.Linear(dense_size, 1)
).to(device)
```

```python
optimizer = torch.optim.Adam(model.parameters())
loss_func = nn.MSELoss()

train_inputs = torch.tensor(splits.train.inputs).float().to(device)
train_labels = torch.tensor(splits.train.labels.values).float().to(device)
train_data = TensorDataset(train_inputs, train_labels)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

val_inputs = torch.tensor(splits.val.inputs).float().to(device)
val_labels = torch.tensor(splits.val.labels.values).float().to(device)
val_data = TensorDataset(val_inputs, val_labels)
val_loader = DataLoader(val_data, batch_size=256)
```

```python
for epoch in range(10):
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        predictions = model(inputs)
        loss = loss_func(predictions, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = []
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            predictions = model(inputs)
            val_loss.append(loss_func(predictions, labels))
        val_loss = np.mean(val_loss)        

    print(f'[E{epoch}] Loss = {loss.item():.3f}. Val loss = {val_loss.item():.3f}')
```

```python
def jointplot(*args, **kwargs):
    ax = sns.jointplot(*args, **kwargs, s=5, alpha=0.3)
    ax.set_axis_labels('Predicted', 'Actual')
```

```python
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        predictions = model(inputs)
        all_preds.append(np.array(predictions))
        all_labels.append(np.array(labels))
    torch.cuda.empty_cache()

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
jointplot(all_preds, all_labels)
```

```python
with torch.no_grad():
    preds = np.array(model(torch.tensor(splits.test.inputs).float().to(device)))
    torch.cuda.empty_cache()

jointplot(preds, splits.test.labels.values)
```

```python
params = []
for name, param in model.named_parameters():
    params.append(np.array(param.detach()))
```

```python
params = np.concatenate([p.ravel() for p in params])
```

```python
np.mean(params), np.std(params), len(params)
```

# Classic BO
 In PDTS, their real-world examples (all?) have a discrete set of
molecules small enough that they can evaluate the network on all of them to find
the predicted best one; it's not clear what they did for the continuous
problems. One might consider sampling points randomly from the space and then
doing a gradient ascent on each; see if, after a certain number of points, you
converge to a (probable) best value.
 
But I think it would be more consistent
to just bin the quantize the points (given that we know a max and min value for
each function). We can make this quite fine, since the ranges are small, and
we're okay with a large number of options. Making each point categorical loses a
lot of information, though, if the functions are smooth at all (which they are).
Ohh, they used a GP not an NN for this section.

```python
import classics
```

```python
func = classics.Bohachevsky()
plot_contours(func, func.bounds);
```

```python
n_hidden = 100
model = nn.Sequential(
    nn.Linear(func.ndim, n_hidden),
    nn.ReLU(),
    nn.Linear(n_hidden, 1)
).to(device)
```

How many points do we need to check in the input space before we get a good idea
of what the model thinks the max is?
  * It seems like not very many (e.g.
1,000), even without doing gradient ascent, but this is certainly problem
specific and might also change as the network is trained.

```python
n_points = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10_000, 25_000, 50_000, 100_000]
max_y = []

for i in range(len(n_points)):
    points = random_points(func.bounds, func.ndim, n_points[i])
    pred_y = model(torch.tensor(points).float().to(device))
    max_y.append(pred_y.max().item())

plt.plot(n_points, max_y)
plt.xlabel('# Points Sampled')
plt.ylabel('Max $\hat Y$')
plt.xscale('log')
plt.rc('font', size=10)
```

What works better - constraining the input every $k$ steps or just once at the
end? What is a good value of $k$?
  * Hard to tell here because the same value
is always the best...

```python
n_points = 1000
n_optimizer_steps = 10000
constrain_steps = [1, 5, 10, 25, 50, 100, 250, 500, 1000, n_optimizer_steps + 1]

points = random_points(func.bounds, func.ndim, n_points)
raw_inputs = torch.tensor(points).float().to(device)

y_maxes = []
x_bests = []

for constrain_every in constrain_steps:
    inputs = raw_inputs.clone().requires_grad_()
    input_optimizer = torch.optim.Adam([inputs])

    optimize_inputs(inputs, input_optimizer, model, n_optimizer_steps, constrain_every)
    
    y_pred = model(inputs)
    y_max = y_pred.max().item()
    x_best = inputs[y_pred.argmax()].detach().cpu().numpy()

    y_maxes.append(y_max)
    x_bests.append(x_best)
```

```python
plt.plot(constrain_steps, y_maxes)
plt.xlabel('Constrain every $k$')
plt.ylabel('Max $\hat y$')
plt.xscale('log')
plt.rc('font', size=15)
```

```python
x_bests
```

## @

```python
prior_mean = 0
prior_std = .05
batch_size = 64
n_points = 500
n_optimizer_steps = 500
constrain_every = 1
```

```python
def get_new_inputs(n_inputs: int, guide, bounds, ndim: int, n_sample_points: int, n_optimizer_steps: int, constrain_every: int):
    new_inputs = []
    for _ in range(n_inputs):
        nn_sample = guide()
        sample_points = random_points(bounds, ndim, n_sample_points)
        sample_points = torch.tensor(sample_points).float().to(device).requires_grad_()
        input_optimizer = torch.optim.Adam([sample_points])

        optimize_inputs(sample_points, input_optimizer, nn_sample, n_optimizer_steps, constrain_every, bounds)

        y_pred = nn_sample(sample_points)
        x_best = sample_points[y_pred.argmax()].detach()
        new_inputs.append(x_best)
    return torch.cat([inputs[None, ...] for inputs in new_inputs])
```

```python
# TODO: can we use an auto guide for this?
bnn_model = make_bnn_model(model, normal_priors(model, prior_mean, prior_std), batch_size=batch_size)
guide = make_guide(model, normal_variationals(model, prior_mean, prior_std))
```

```python
def eval_on_grid(guide, bounds, device: torch.device, n_samples: int=50, n_points: int=100):
    xs = np.linspace(*bounds[0], n_points)
    ys = np.linspace(*bounds[1], n_points)
    xs, ys = np.meshgrid(xs, ys)
    grid = np.stack((xs, ys), axis=-1)
    grid = torch.tensor(grid).float().to(device)

    preds = []
    for _ in range(n_samples):
        nn_sample = guide()
        preds.append(nn_sample(grid).detach().cpu().squeeze().numpy())
    return np.stack(preds, axis=-1).mean(axis=-1)
```

```python
ax = plot_contours([func, eval_on_grid(guide, func.bounds, device)], func.bounds, titles=['True', 'Predicted'])
```

```python
# for the first batch, select the points at random from the space
inputs = random_points(func.bounds, func.ndim, n_points=batch_size)
labels = func(inputs)

inputs = torch.tensor(inputs).float().to(device)
labels = torch.tensor(labels).float().to(device)
```

```python
p = guide()(inputs)
((p - labels) ** 2).sum()
```

```python
optimizer = pyro.optim.Adam({})
pyro.clear_param_store()
svi = pyro.infer.SVI(bnn_model, guide, optimizer, loss=pyro.infer.Trace_ELBO())

losses = []
```

```python
losses += train(10000, inputs, labels, verbose=True)
```

```python
plot_contours([func, eval_on_grid(guide, func.bounds, device)], func.bounds, titles=['True', 'Predicted'])
```

```python
inputs = get_new_inputs(batch_size, guide, func.bounds, func.ndim, n_points, n_optimizer_steps, constrain_every)

labels = func(inputs.cpu().numpy())
labels = torch.tensor(labels).float().to(device)
```

```python
inputs = random_points(func.bounds, func.ndim, n_points=batch_size)

labels = (inputs ** 2).mean(axis=1)
# labels = func(inputs)

inputs = torch.tensor(inputs).float().to(device)
labels = torch.tensor(labels).float().to(device)
```

```python
n_hidden = 256
nn_sample = nn.Sequential(
  nn.Linear(in_features=2, out_features=n_hidden),
  nn.Tanh(),
  nn.Linear(in_features=n_hidden, out_features=1)
)
```

```python
optimizer = torch.optim.Adam(nn_sample.parameters())
loss_func = nn.MSELoss()
```

```python
for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(inputs).squeeze()
    loss = loss_func(predictions, labels)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'[E{epoch}] Loss = {loss.item():,.3f}.')
```

```python
def jointplot(*args, **kwargs):
    ax = sns.jointplot(*args, **kwargs, s=5, alpha=0.3)
    ax.set_axis_labels('Predicted', 'Actual')
```

```python
preds = model(inputs).detach().cpu().squeeze().numpy()
jointplot(preds, labels.cpu().numpy())
```

```python
params = []
for name, param in model.named_parameters():
    params.append(np.array(param.detach()))
```

```python
params = np.concatenate([p.ravel() for p in params])
```

```python
np.mean(params), np.std(params), len(params)
```

# BNN

```python
prior_mean = 0
prior_std = .05
batch_size = 128
```

```python
def get_corr(guide, inputs, labels, n_samples: int, plot: bool=False):
    all_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            nn_sample = guide()  # guide doesn't need input data
            nn_sample.eval()
            all_preds.append(np.array(nn_sample(inputs)))
    all_preds = np.concatenate(all_preds, axis=1)
    mean_preds = all_preds.mean(axis=1)
    
    if plot:
        jointplot(mean_preds, labels)

    corr = pearsonr(mean_preds, labels)[0]
    return corr

def train(n_steps: int, verbose: bool=False):
    losses = []
    train_corrs = []
    val_corrs = []

    for step in range(n_steps):
        loss = svi.step(train_inputs, train_labels)
        losses.append(loss)
        if step % 500 == 0:
            train_corr = get_corr(guide, train_inputs, splits.train.labels.values.squeeze(), n_samples=50)
            val_corr = get_corr(guide, val_inputs, splits.val.labels.values.squeeze(), n_samples=50)
            train_corrs.append(train_corr)
            val_corrs.append(val_corr)
            
            if verbose:
                print(f"[S{step:04}] loss: {loss:,.0f}; t_corr: {train_corr:.3f}; v_corr: {val_corr:.3f}")
    return losses, train_corrs, val_corrs
```

```python
optimizer = pyro.optim.Adam({})

priors = lambda: normal_priors(model, prior_mean, prior_std)
variational_dists = lambda: normal_variationals(model, prior_mean, prior_std)

bnn_model = make_bnn_model(model, priors, batch_size=batch_size)
guide = make_guide(model, variational_dists)
```

```python
pyro.clear_param_store()
svi = pyro.infer.SVI(bnn_model, guide, optimizer, loss=pyro.infer.Trace_ELBO())

losses = []
train_corrs = []
val_corrs = []
```

```python
res = train(40000, verbose=True)
losses += res[0]
train_corrs += res[1]
val_corrs += res[2]
```

```python
plt.plot(losses)
```

```python
plt.plot(losses[-5000:])
```

There never seems to be any overfitting, which is nice; we can train the model
as long as we want (until the validation performance hasn't improved for some
time?).

```python
plt.plot(train_corrs, label='train')
plt.plot(val_corrs, label='val')
plt.legend()
```

```python
plt.plot(abs(np.array(train_corrs) - np.array(val_corrs)))
```

### Optimizing Input

Should I try using the ProtVecs as input instead? That
space is continuous, at least.
  * For decoding, do this in a smart way that
uses the overlap of the kmers; instead of just decoding the first vector to a
3-mer, consider how compatible the two residues that overlap with the 3-mer for
the second vector are. E.g. if the first vector is closest to SRG but the second
is closest to AGT, and the 3rd is closest to SQW, we have
  SRG
   AGT
    SQW
and we need to decide what we do at the locations that don't agree.
  * I think
we should formulate the decoding as a dynamic programming problem somehow; try
to find the sequence whose total (Euclidean) distance (when embedded as a
protvec) is closest to the actual protvecs we have (instead of doing a greedy
decoding).
* Maybe compare one-hot vs protvec (and also different decoding
methods for each? e.g. project back to seq space every step for one-hot vs
project back every $k$ vs only at the end) in terms of the predicted affinity of
the sequence we get after decoding the optimized input

```python
nn_sample = guide()
nn_sample
```

```python
inputs = train_inputs[1:5].clone().requires_grad_()
input_opt = torch.optim.Adam([inputs])
```

```python
def update_inp():
    # TODO: constrain input in some way here
    input_opt.zero_grad()
    output = nn_sample(inp)
    # don't know if we should need `retain_graph`; there might
    # be something better I should be doing... 
    (-output).backward(torch.ones_like(output))
    return output
```

```python
inp.grad
```

```python
input_opt.zero_grad()
```

```python
output = nn_sample(inp)
# don't know if we should need `retain_graph`; there might
# be something better I should be doing... 
(-output).backward(torch.ones_like(output))
```

```python
inp.grad.shape
```

```python
outputs = []
for _ in range(10000):
    outputs.append(input_opt.step(update_inp).detach().cpu().numpy())
outputs = np.concatenate(outputs, axis=1)
```

```python
plt.plot(outputs[0])
```

```python
inp[0:1].argmax(dim=1)
```

```python
inp.argmax()
```

### Variance vs # NN Samples

How does the variance in the predictive
distribution for a given input change as we use more models?
  * It seems that
the mean variance first goes up (which you expect because there can't be much
variance with only a couple of models) and then levels off (at the "true" value,
or at least what we think it is).
  * 500 models seem sufficient to get the
proper variance reproducibly.

```python
n_samples = [2, 5, 10, 25, 50, 100, 250, 500, 1000, 1500, 2000, 5000]
mean_vars = []
n_repeats = 3

with torch.no_grad():
    for n in n_samples:
        mean_var = []
        for rep in range(n_repeats):
            preds = []
            for _ in range(n):
                nn_sample = guide()
                nn_sample.eval()
                preds.append(np.array(nn_sample(val_inputs)))
            preds = np.array(preds).squeeze().T
            mean_var.append(preds.var(axis=1).mean())
        mean_vars.append(mean_var)
```

```python
plt.figure(figsize=(12, 8))
plt.rc('font', size=18)
plt.scatter(n_samples, [np.mean(repeats) for repeats in mean_vars], s=30)
plt.plot(n_samples, [np.mean(repeats) for repeats in mean_vars])
plt.errorbar(n_samples, [np.mean(repeats) for repeats in mean_vars],
             [2 * np.std(repeats) for repeats in mean_vars],
             linestyle="None", c='red')
plt.xscale('log')
plt.xlabel('# NN Samples')
plt.ylabel('Mean Variance')
plt.title('Expected Predictive Distribution Variance');
```

### Posterior Predictive Distributions

Are the posterior predictive
distributions always monotonic?
  * Yes, that seems to be the case.

```python
preds = []
with torch.no_grad():
    for _ in range(500):
        nn_sample = guide()
        nn_sample.eval()
        preds.append(np.array(nn_sample(val_inputs)))
preds = np.array(preds).squeeze().T
```

```python
plt.figure(figsize=(6, 4))
[sns.distplot(preds[i]) for i in np.random.randint(len(preds), size=5)];
```

```python
# you can also look at the overall affinity distribution of each sampled model
# and see how those compare

[sns.distplot(preds.T[i]) for i in np.random.randint(len(preds.T), size=20)];
```

# Model Saving / Loading Testing
Make convenience functions for these things;
probably start some pytorch utils file.

```python
ps = pyro.get_param_store()
ps
```

```python
ps.save('models/params')
```

```python
optimizer.save('models/opt')
```

```python
pyro.module()
```

```python
pyro.random_module()
```

```python
ps.save()
```

```python
save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'model_state': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer_state' : optimizer.state_dict(),
        }, is_best)

def save_checkpoint(state, is_best, fname='checkpoint.pth.tar'):
    torch.save(state, fname)
    if is_best:
        shutil.copyfile(fname, 'model_best.pth.tar')

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
```
