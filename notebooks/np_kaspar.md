```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Callable

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

device = torch.device('cuda:0' if torch.has_cudnn else 'cpu')
%matplotlib inline
```

```python
# f(x, y) = r
class Representer(nn.Module):
    def __init__(self, x_y_dim, hidden_dim, r_dim):
        super().__init__()

        self.fc1 = nn.Linear(x_y_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, r_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_y):
        hidden = self.sigmoid(self.fc1(x_y))
        hidden = self.sigmoid(self.fc2(hidden))
        r_i = self.fc3(hidden)
        return r_i.mean(dim=0)
```

```python
# q(z|r) = q(z|x, y)
class ZDistribution(nn.Module):
    def __init__(self, r_dim, hidden_dim, z_dim):
        super().__init__()

        self.fc1 = nn.Linear(r_dim, z_dim)
        self.fc2 = nn.Linear(r_dim, z_dim)

        self.softplus = nn.Softplus()

    def forward(self, r):
        z_loc = self.fc1(r)
        z_scale = self.softplus(self.fc2(r))
        return z_loc, z_scale
```

```python
# y_mean = f(z, x)
class Predictor(nn.Module):
    def __init__(self, z_x_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(z_x_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_x):
        hidden = self.sigmoid(self.fc1(z_x))
        hidden = self.sigmoid(self.fc2(hidden))
        hidden = self.sigmoid(self.fc3(hidden))
        hidden = self.sigmoid(self.fc4(hidden))
        y_mean = self.fc5(hidden)
        return y_mean
```

```python
class NP(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int=8, r_dim: int=2, z_dim: int=2):
        super().__init__()

        self.representer = Representer(x_dim + y_dim, hidden_dim, r_dim)
        self.z_distribution = ZDistribution(r_dim, hidden_dim, z_dim)
        self.predictor = Predictor(z_dim + x_dim, hidden_dim, y_dim)
        self.n_samples = 30

    def forward(self, n_context, x_all, y_all, train: bool=True):
        y_scale = 0.1

        x_context = x_all[:n_context]
        y_context = y_all[:n_context]
        x_target = x_all[n_context:]
        y_target = y_all[n_context:]

        # the prior is q(z|x_C, y_C)
        r_context = self.representer(torch.cat((x_context, y_context), dim=1))
        r_all = self.representer(torch.cat((x_all, y_all), dim=1))

        z_loc_context, z_scale_context = self.z_distribution(r_context)
        z_loc_all, z_scale_all = self.z_distribution(r_all)
        
        z_normal_all = torch.distributions.Normal(z_loc_all, z_scale_all)
        z_normal_context = torch.distributions.Normal(z_loc_context, z_scale_context)
        z = z_normal_all.sample((self.n_samples,))
        
        x = x_target if train else x_all
        
        # broadcast z and x to the same shape[:-1] : (n_samples, batch_size, n_features)
        batch_size = len(x)
        z = z.view(self.n_samples, 1, -1).expand(-1, batch_size, -1)
        x = x.view(1, batch_size, -1).expand(self.n_samples, -1, -1)

        z_x = torch.cat((z, x), dim=-1)
        y_mean = self.predictor(z_x)

        kl = torch.distributions.kl_divergence(z_normal_all, z_normal_context).sum()
        
        return y_mean, y_scale, kl
```

```python
def compute_log_likelihood(y_loc, y_scale, y_target):
    normal = torch.distributions.Normal(y_loc, y_scale)
    # sum over points, average over samples from z
    return normal.log_prob(y_target).sum(dim=-1).mean()
```

# Kaspar's 1st Experiment
* Note that the exact architecture of the modules
above may be different if they've been changed for experimenting with other
tasks

```python
x = np.array([[-2, -1, 0, 1, 2]]).T
y = np.sin(x)
plt.scatter(x, y)

x = torch.tensor(x).float().to(device)
y = torch.tensor(y).float().to(device)
```

```python
x_dim = 1
y_dim = 1
learning_rate = 1e-3
hidden_dim = 8
r_dim = 2
z_dim = 2

nprocess = NP(x_dim, y_dim, hidden_dim, r_dim, z_dim).to(device)
optimizer = torch.optim.Adam(nprocess.parameters(), learning_rate)
```

```python
n_steps = 10_000

shuffle_idx = list(range(len(x)))

losses = []
kls = []
log_likelihoods = []
for step in range(n_steps):
    optimizer.zero_grad()

    np.random.shuffle(shuffle_idx)
    n_context = np.random.randint(1, len(x))    

    y_loc, y_scale, kl = nprocess(n_context, x[shuffle_idx], y[shuffle_idx])

    y_target = y[shuffle_idx][n_context:]
    log_likelihood = compute_log_likelihood(y_loc, y_scale, y_target)
    loss = -log_likelihood + kl
    
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    kls.append(kl.item())
    log_likelihoods.append(log_likelihood.item())
    if step % 500 == 0:
        print(f"{step}: {np.mean(losses[-500:]):,.3f}")
```

```python
(pd.DataFrame({'loss': losses, 'kl': kls, '-log p': -np.array(log_likelihoods)})
   .rolling(window=20).mean().plot()
)
plt.legend()
```

```python
x_plot = torch.cat((x, torch.linspace(-2, 2, 100).to(device).view(-1, 1)), dim=0)
y_plot = torch.sin(x_plot)

y_loc, y_scale, _ = nprocess(5, x_plot, y_plot)
x_plot = x_plot[5:].cpu().numpy().ravel()
y_plot = y_plot[5:].cpu().numpy().ravel()
y_loc = y_loc[0].detach().cpu().numpy().ravel()

plt.plot(x_plot, y_loc, label='Predicted')
plt.fill_between(x_plot, y_loc - y_scale, y_loc + y_scale, alpha=0.5)
plt.plot(x_plot, y_plot, label='True')
plt.scatter(x, y, label='Observed', c='green', marker='+')
plt.legend()
```

# MNIST Image Completion

```python
data_dir = '../data'
train_set = MNIST(data_dir, train=True, transform=transforms.ToTensor(), download=True)
train_imgs = train_set.train_data.float() / 255

test_set = MNIST(data_dir, train=False, transform=transforms.ToTensor(), download=True)
test_imgs = test_set.test_data.float() / 255
```

```python
x_dim = 2
y_dim = 1
learning_rate = 1e-3
hidden_dim = 128
r_dim = 128
z_dim = 128

nprocess = NP(x_dim, y_dim, hidden_dim, r_dim, z_dim).to(device)
optimizer = torch.optim.Adam(nprocess.parameters(), learning_rate)
```

```python
n_steps = 3 * len(train_imgs)  # I usually interrupt it before this finishes

# the pixel locations are the same for all images
width, height = train_imgs.shape[1:3]
x = torch.tensor([[i / (height - 1), j / (width - 1)] for i in range(height) for j in range(width)])
x = x.to(device)

shuffle_idx = list(range(len(x)))

losses = []
kls = []
log_likelihoods = []
for step in range(n_steps):
    optimizer.zero_grad()
    img_idx = np.random.randint(len(train_imgs))

    y = train_imgs[img_idx].reshape(-1, 1).to(device)

    np.random.shuffle(shuffle_idx)
    n_context = np.random.randint(1, len(x) // 2)
    
    y_loc, y_scale, kl = nprocess(n_context, x[shuffle_idx], y[shuffle_idx])

    y_target = y[shuffle_idx][n_context:]
    log_likelihood = compute_log_likelihood(y_loc, y_scale, y_target)
    loss = -log_likelihood + kl
    
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    kls.append(kl.item())
    log_likelihoods.append(log_likelihood.item())
    if step % 1000 == 0:
        print(f"{step}: {np.mean(losses[-1000:]):,.2f}")
```

```python
(pd.DataFrame({'loss': losses, 'kl': kls, '-log p': -np.array(log_likelihoods)})
   .rolling(window=200).mean().plot()
)
plt.legend()
```

```python
def plot_samples(model: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor], n_contexts: Tuple[int]=(10, 100, 300, 784), n_preds: int=3):
    img_idx = np.random.randint(len(test_imgs))
    y = test_imgs[img_idx].reshape(-1, 1).to(device)

    np.random.shuffle(shuffle_idx)
    unshuffle_idx = np.argsort(shuffle_idx)

    fig, subplots = plt.subplots(figsize=(10, 10), nrows=n_preds + 1, ncols=len(n_contexts))

    # n_context = np.random.randint(1, len(x) // 2)
    for (context_i, n_context) in enumerate(n_contexts):
        subplots[0, context_i].imshow(test_imgs[img_idx], cmap='Greys')
        subplots[0, context_i].axis('off')
        subplots[0, context_i].set_title(f'{n_context} context')

        for i in range(n_preds):
            pred = model(n_context, x[shuffle_idx], y[shuffle_idx])

            try:
                pred = pred.detach()
            except AttributeError:  # not a torch.Tensor
                pass
#             subplots[i + 1, context_i].imshow(pred[unshuffle_idx].view((width, height, 1)).expand(-1, -1, 3))
            subplots[i + 1, context_i].imshow(pred[unshuffle_idx].reshape(width, height), cmap='Greys')
            subplots[i + 1, context_i].axis('off')
```

```python
def get_pixel_idx(img) -> torch.Tensor:
    height, width = img.shape
    idx = torch.tensor([[i / (height - 1), j / (width - 1)] for i in range(height) for j in range(width)])
    return idx
```

```python
def predict(img: torch.Tensor, n_context: int=100) -> torch.Tensor:
    height, width = img.shape
    pixel_idx = get_pixel_idx(img).to(device)
    img = img.reshape(-1, 1).to(device)
    
    shuffle_idx = list(range(width * height))
    np.random.shuffle(shuffle_idx)
    unshuffle_idx = np.argsort(shuffle_idx)
    
    pred, *_ = nprocess(n_context, pixel_idx[shuffle_idx], img[shuffle_idx], train=False)
    return pred[0][unshuffle_idx].reshape(height, width).detach()


def plot_img_pred(pred):
    plt.figure(figsize=(6, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(pred.view((len(pred), -1, 1)).expand(-1, -1, 3))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(pred, cmap='Greys')
    plt.axis('off')
```

```python
pred = predict(test_imgs[0])
plot_img_pred(pred)
```

```python
plot_samples(lambda *args: nprocess(*args, train=False)[0][0])
```
