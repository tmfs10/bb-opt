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
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, r_dim)
        self.activation = nn.Softplus()

    def forward(self, x_y):
        hidden = self.activation(self.fc1(x_y))
        hidden = self.activation(self.fc2(hidden))
        hidden = self.activation(self.fc3(hidden))
        r_i = self.fc4(hidden)
        return r_i.mean(dim=0)
```

```python
# q(z|r) = q(z|x, y)
class ZDistribution(nn.Module):
    def __init__(self, r_dim, hidden_dim, z_dim):
        super().__init__()

        self.fc1 = nn.Linear(r_dim, hidden_dim)
        self.fc2a = nn.Linear(hidden_dim, z_dim)
        self.fc2b = nn.Linear(hidden_dim, z_dim)

        self.activation = nn.Softplus()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, r):
        hidden = self.activation(self.fc1(r))
        z_loc = self.fc2a(hidden)
#         z_scale = self.softplus(self.fc2b(hidden))
        z_scale = self.sigmoid(self.fc2b(hidden))

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
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Softplus()

    def forward(self, z_x):
        hidden = self.activation(self.fc1(z_x))
        hidden = self.activation(self.fc2(hidden))
        hidden = self.activation(self.fc3(hidden))
        hidden = self.activation(self.fc4(hidden))
        hidden = self.activation(self.fc5(hidden))
        y_mean = self.fc6(hidden)
        return y_mean
```

```python
class NP(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int=8, r_dim: int=2, z_dim: int=2):
        super().__init__()

        self.representer = Representer(x_dim + y_dim, hidden_dim, r_dim)
        self.z_distribution = ZDistribution(r_dim, hidden_dim, z_dim)
        self.predictor = Predictor(z_dim + x_dim, hidden_dim, y_dim)
        self.n_samples = 1

    def forward(self, n_context, x_all, y_all, cnp_mode, train: bool=True, return_z=False):
        y_scale = 0.05

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

#         fixed_sigma = torch.tensor(1.).to(device)
#         z_normal_all = torch.distributions.Normal(z_loc_all, fixed_sigma)
#         z_normal_context = torch.distributions.Normal(z_loc_context, fixed_sigma)
        
#         z = z_normal_all.sample((self.n_samples,))

#         if img_label == 3:
#             z = torch.tensor(0.).to(device).view(1, 1).expand(self.n_samples, -1)
#         else:
#             z = torch.tensor(1.).to(device).view(1, 1).expand(self.n_samples, -1)
    
#         z = r_context.view(-1, 1).expand(self.n_samples, -1)
        
        if cnp_mode:
            z = z_loc_all.view(-1, 1).expand(self.n_samples, -1)
        else:
            z = z_normal_all.sample((self.n_samples,))
    
        if return_z:
            return z_loc_all, z_scale_all, z_loc_context, z_scale_context
        
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
x_min = -2
x_max = 2
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
n_steps = 5_000

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
train_3 = train_imgs[train_set.train_labels == 3]
train_5 = train_imgs[train_set.train_labels == 5]

train_imgs = torch.cat((train_3[0:1], train_5[0:1])) 
train_imgs.shape
```

```python
x_dim = 2
y_dim = 1
learning_rate = 1e-3
hidden_dim = 64
r_dim = 1
z_dim = 1

nprocess = NP(x_dim, y_dim, hidden_dim, r_dim, z_dim).to(device)
optimizer = torch.optim.Adam(nprocess.parameters(), learning_rate)
```

```python
# the pixel locations are the same for all images
width, height = train_imgs.shape[1:3]
x = torch.tensor([[i / (height - 1), j / (width - 1)] for i in range(height) for j in range(width)])
x = x.to(device)

shuffle_idx = list(range(len(x)))

losses = []
kls = []
log_likelihoods = []
```

```python
# n_steps = 3 * len(train_imgs)  # I usually interrupt it before this finishes
n_steps = 10_000
n_imgs = 2
cnp_mode = False
nprocess.n_samples = 1000

for step in range(n_steps):
    optimizer.zero_grad()
    
    loss = torch.tensor(0.).to(device)
    
    for img_idx in [0, 1]:
        y = train_imgs[img_idx].reshape(-1, 1).to(device)

        np.random.shuffle(shuffle_idx)
    #     n_context = np.random.randint(1, len(x) // 2)
        n_context = len(x) // 2
        
        y_loc, y_scale, kl = nprocess(n_context, x[shuffle_idx], y[shuffle_idx], cnp_mode=cnp_mode)

        y_target = y[shuffle_idx][n_context:]
        log_likelihood = compute_log_likelihood(y_loc, y_scale, y_target)
        loss += -log_likelihood
        
        if not cnp_mode:
            loss += kl

    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    kls.append(kl.item())
    log_likelihoods.append(log_likelihood.item())
    if step % 1000 == 0:
        print(f"{step}: {np.mean(losses[-1000:]):,.2f}")
```

```python
# n_steps = 3 * len(train_imgs)  # I usually interrupt it before this finishes
n_steps = 10_000

for step in range(n_steps):
    optimizer.zero_grad()
    img_idx = np.random.randint(len(train_imgs))

    y = train_imgs[img_idx].reshape(-1, 1).to(device)

    np.random.shuffle(shuffle_idx)
    n_context = np.random.randint(1, len(x))
    
    y_loc, y_scale, kl = nprocess(n_context, x[shuffle_idx], y[shuffle_idx])

    y_target = y[shuffle_idx][n_context:]
    log_likelihood = compute_log_likelihood(y_loc, y_scale, y_target)
    loss = -log_likelihood# + kl

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
   .rolling(window=20).mean().plot()
)
plt.legend()
```

```python
# cnp_mode = True
n_context = 350

plot_img_pred(train_imgs[0], n_context=1, cnp_mode=cnp_mode)
```

```python
plot_img_pred(train_imgs[1], n_context=n_context, cnp_mode=cnp_mode)
```

```python
r3_context = nprocess.representer(torch.cat((x[:n_context],
                                             train_imgs[0].reshape(-1, 1).to(device)[:n_context]), dim=1))
r3_context
```

```python
r3 = nprocess.representer(torch.cat((x, train_imgs[0].reshape(-1, 1).to(device)), dim=1))
r3
```

```python
nprocess.z_distribution(r3_context)
```

```python
nprocess.z_distribution(r3)
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
        subplots[0, context_i].imshow(test_imgs[img_idx], cmap='Greys_r')
        subplots[0, context_i].axis('off')
        subplots[0, context_i].set_title(f'{n_context} context')

        for i in range(n_preds):
            pred = model(n_context, x[shuffle_idx], y[shuffle_idx])

            try:
                pred = pred.detach()
            except AttributeError:  # not a torch.Tensor
                pass
#             subplots[i + 1, context_i].imshow(pred[unshuffle_idx].view((width, height, 1)).expand(-1, -1, 3))
            subplots[i + 1, context_i].imshow(pred[unshuffle_idx].reshape(width, height), cmap='Greys_r')
            subplots[i + 1, context_i].axis('off')
```

```python
def get_pixel_idx(img) -> torch.Tensor:
    height, width = img.shape
    idx = torch.tensor([[i / (height - 1), j / (width - 1)] for i in range(height) for j in range(width)])
    return idx
```

```python
def predict(img: torch.Tensor, n_context: int=100, cnp_mode=False) -> torch.Tensor:
    height, width = img.shape
    pixel_idx = get_pixel_idx(img).to(device)
    img = img.reshape(-1, 1).to(device)
    
    shuffle_idx = list(range(width * height))
    np.random.shuffle(shuffle_idx)
    unshuffle_idx = np.argsort(shuffle_idx)
    
    pred, *_ = nprocess(n_context, pixel_idx[shuffle_idx], img[shuffle_idx], cnp_mode, train=False)
    return pred[0][unshuffle_idx].reshape(height, width).detach()


def plot_img_pred(img, n_context: int=100, cnp_mode=False):
    pred = predict(img, n_context, cnp_mode)

    plt.figure(figsize=(6, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(pred.view((len(pred), -1, 1)).expand(-1, -1, 3).clamp(0, 1))
    plt.axis('off')
    plt.title('Predicted')

    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap='Greys_r')
    plt.axis('off')
    plt.title(f'True (# context = {n_context})')
```

```python
plot_img_pred(train_imgs[4], n_context=700)
```

```python
plot_img_pred(train_imgs[0])
```

```python
plot_samples(lambda *args: nprocess(*args, train=False)[0][0])
```
