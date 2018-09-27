
import numpy as np
from scipy.stats import kendalltau

import hsic
import torch
import torch.nn as nn

from chemvae_keras import vae_utils
from chemvae_keras import mol_utils as mu

import utils
import reparam_trainer as reparam
from tqdm import tnrange


def get_activation(act):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU
    elif act == 'sigmoid':
        return nn.Sigmoid
    elif act == 'tanh':
        return nn.Tanh

class PropertyPredictor(nn.Module):

    def __init__(self, params):
        self.params = params
        activation = get_activation(params.prop_activation)

        num_inputs = params.prop_pred_num_input_features + params.prop_pred_num_random_inputs
        net = [nn.Linear(num_inputs, params.prop_pred_num_hidden), activation()]
        if params.prop_pred_dropout > 0:
            net += [nn.Dropout(params.prop_pred_dropout)]

        if params.prop_pred_depth > 1:
            num_inputs = params.prop_pred_num_hidden
            for p_i in range(1, params.prop_pred_depth):
                num_outputs = int(params.prop_pred_num_hidden * (params.prop_pred_growth_factor**p_i))
                prop_mid = nn.Linear(num_inputs, num_outputs)

                net += [prop_mid, activation()]
                if params.prop_pred_dropout > 0:
                    net += [nn.Dropout(params.prop_pred_dropout)]
                if params.prop_batchnorm:
                    net += [nn.BatchNorm1d(num_outputs)]

                num_inputs = num_outputs

        net += [nn.Linear(num_inputs, 3)]
        self.net = nn.ModuleList(net)

    def forward(self, x, z, resize_at_end=False):
        assert x.ndimension() == 2
        num_samples = z.shape[0]
        N = x.shape[0]

        x = utils.collated_expand(x, num_samples)
        z = z.repeat([N, 1])
        x = torch.cat([x, z], dim=1)

        for h in self.net:
            x = h(x)

        if resize_at_end:
            x = x.view([N, num_samples]).transpose()
        return x

    def input_shape(self):
        return self.params.prop_pred_num_input_features


def train(
        params,
        data,
        model,
        qz,
        e_dist,
):
    losses = []
    kl_losses = []
    hsic_losses = []
    val_losses = []

    corrs = []
    val_corrs = []

    train_X, train_Y, val_X, val_Y = data

    N = train_inputs.shape[0]
    num_batches = N//params.train_batch_size

    model_parameters = []
    for m in [model, qz]:
        model_parameters += list(m.parameters())
    batches, optim = reparam.init_train(params, model_parameters, train_inputs, train_labels)

    progress = tnrange(params.num_epochs)

    for epoch_iter in progress:
        for bi in range(num_batches):
            bs = batches[bi]
            be = batches[bi+1]
            bN = be-bs

            bX = train_X[bs:be]
            bY = train_Y[bs:be]

            for k in range(1):
                e = reparam.generate_prior_samples(params.num_samples, e_dist)
                loss, log_prob_loss, kl_loss, hsic_loss, _, _ = reparam.compute_loss(params, bX, bY, model, qz, e, hsic_lambda=params.hsic_train_lambda)

                losses += [log_prob_loss]
                kl_losses += [kl_loss]
                hsic_losses += [hsic_loss]

                optim.zero_grad()
                loss.backward()
                optim.step()

        e = reparam.generate_prior_samples(params.num_samples, e_dist)
        preds = reparam.predict(train_X, model, qz, e)[:, :, 0].mean(1)
        assert preds.shape == train_labels.shape, str(preds.shape) + " == " + str(train_labels.shape)

        corrs.append(kendalltau(preds, train_labels)[0])

        preds = reparam.predict(val_X, model, qz, e)[:, :, 0].mean(1)
        assert preds.shape == val_labels.shape

        val_corr = kendalltau(preds, val_labels)[0]

        val_corrs.append(val_corr)
        progress.set_description(f"Corr: {val_corr:.3f}")
        progress.set_postfix({'hsic_loss' : hsic_losses[-1], 'kl_loss' : kl_losses[-1], 'log_prob_loss' : losses[-1]})

    return losses, kl_losses, hsic_losses, val_losses, corrs, val_corrs


def acquire_properties(encoded, vae, device='cuda'):
    batch_size = encoded.shape[0]
    decoded = vae.decode(encoded)
    smiles = vae.hot_to_smiles(decoded, strip=True)

    props = torch.zeros([batch_size, 3], device=device)
    assert len(smiles) == batch_size
    for i in range(batch_size):
        props[i][:] = utils.all_scores(smiles[i])
    return props


def load_zinc250k(num_to_load=0, score_fn=None, batch_size=128):
    filename = '/cluster/sj1/bb_opt/chemical_vae/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv'
    zinc250k = [[], []]
    with open(filename) as f:
        next(f)
        for line in f:
            line = [k.strip() for k in line.strip().split('\t')]
            zinc250k[0] += [line[0]]
            zinc250k[1] += [[float(k) for k in line[1:]]]
    zinc250k[1] = np.array(zinc250k[1], dtype=np.float32)
    assert zinc250k[1].shape[1] == 3
    labels = np.array([score_fn(k) for k in zinc250k[1]])

    if score_fn is not None:
        sort_idx = np.argsort(labels)
        zinc250k[0] = [zinc250k[0][i] for i in sort_idx]
        zinc250k[1] = zinc250k[1][sort_idx, :]

    if num_to_load > 0:
        zinc250k[0] = zinc250k[0][:num_to_load]
        zinc250k[1] = zinc250k[1][:num_to_load, :]
    num_total = len(zinc250k[0])

    print("read zinc250k from file")

    zinc_prop_dir = '/cluster/sj1/bb_opt/chemical_vae/models/zinc_properties'
    vae = vae_utils.VAEUtils(directory=zinc_prop_dir)

    print("initialized VAEUtils")

    smiles_one_hot = []
    for i in range(len(zinc250k[0])):
        smiles_string = zinc250k[0][i]
        props = zinc250k[1][i]
        if i % 10000 == 0:
            print("done {:d}K samples".format(i//1000))
        smiles_one_hot += [vae.smiles_to_hot(mu.canon_smiles(smiles_string), canonize_smiles=True)[0]]
    smiles_one_hot = np.array(smiles_one_hot)
    assert len(smiles_one_hot.shape) == 3, str(smiles_one_hot.shape)
    assert smiles_one_hot.shape[0] == num_total, str(smiles_one_hot.shape) + "[0] == " + str(num_total)

    print("converted to one_hot")

    smiles_z = []
    bs = 0
    while bs < num_total:
        be = max(bs+batch_size, num_total)
        smiles_z += [vae.encode(smiles_one_hot[bs:be])]
        bs = be
    smiles_z = np.concatenate(smiles_z, axis=0)
    assert len(smiles_z.shape) == 2, str(smiles_z.shape)
    print(smiles_z.shape)

    print("processed encoded representation")

    return zinc250k, vae, smiles_one_hot, smiles_z, labels

def load_zinc(num_to_load=0):
    filename = '/cluster/sj1/bb_opt/data/zinc/10_p0.smi'

    zinc = []
    with open(filename) as f:
        for line in f:
            if num_to_load > 0 and len(zinc) >= num_to_load:
                break
            line = line.strip()
            zinc += [line]

    return zinc
