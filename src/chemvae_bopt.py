
import tensorflow as tf
import keras
import numpy as np

import hsic
import torch
import torch.nn as nn

from chemvae_keras import vae_utils
from chemvae_keras import mol_utils as mu

import utils


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

def load_zinc250k(score_fn = lambda x : x[0]):
    filename = '/cluster/sj1/bb_opt/chemical_vae/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv'
    zinc250k = [[], []]
    with open(filename) as f:
        next(f)
        for line in f:
            line = [k.strip() for k in line.strip().split('\t')]
            zinc250k[0] += [line[0]]
            zinc250k[1] += [[float(k) for k in line[1:]]]
    zinc250k[1] = np.array(zinc250k, dtype=np.float32)

    zinc_prop_dir = '/cluster/sj1/bb_opt/chemical_vae/models/zinc_properties'
    vae = vae_utils.VAEUtils(directory=directory)

    sort_idx = np.argsort([score_fn(k) for k in zinc250k[1]])
    zinc250k[0] = [zinc250k[0][i] for i in sort_idx]
    zinc250k[1] = zinc250k[1][sort_idx, :]

    smiles_one_hot = []
    for i in range(len(zinc250k[0])):
        smiles_string = zinc250k[0][i]
        props = zinc250k[1][i]
        if i % 10000 == 0:
            print("done {:d}K samples".format(i//1000))
        smiles_one_hot += [vae.smiles_to_hot(mu.canon_smiles(smiles_string), canonize_smiles=True)]

    smiles_z = [vae.encode(k)[0] for k in smiles_one_hot]

    return zinc250k, vae, smiles_one_hot, smiles_z

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
