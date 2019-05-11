
import numpy as np
from scipy.stats import kendalltau

import hsic
import torch
import torch.nn as nn

import non_matplotlib_utils as utils
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

class PropertyPredictorPaper(nn.Module):

    def __init__(self, params):
        super(PropertyPredictorPaper, self).__init__()
        self.params = params
        activation = get_activation(params.prop_activation)

        num_inputs = params.prop_pred_num_input_features + params.prop_pred_num_random_inputs
        net = [nn.Linear(num_inputs, 1000), activation()]
        net += [nn.Dropout(0.2)]
        net += [nn.Linear(1000, 1000), activation()]
        net += [nn.Dropout(0.2)]
        net += [nn.Linear(1000, 1)]

        self.net = nn.ModuleList(net)

    def forward(self, x, z):
        x = torch.cat([x, z], dim=1)
        for h in self.net:
            x = h(x)
        return x

    def input_shape(self):
        return [self.params.prop_pred_num_input_features]


class PropertyPredictor(nn.Module):

    def __init__(
            self, 
            params,
            num_inputs, 
            ):
        super(PropertyPredictor, self).__init__()
        self.params = params
        activation = get_activation(params.chemvae_prop_activation)

        net = [nn.Linear(num_inputs, params.chemvae_prop_pred_num_hidden), activation()]
        if params.chemvae_prop_pred_dropout > 0:
            net += [nn.Dropout(params.chemvae_prop_pred_dropout)]

        if params.chemvae_prop_pred_depth > 1:
            num_inputs = params.chemvae_prop_pred_num_hidden
            for p_i in range(1, params.chemvae_prop_pred_depth):
                num_outputs = int(params.chemvae_prop_pred_num_hidden * (params.chemvae_prop_pred_growth_factor**p_i))
                prop_mid = nn.Linear(num_inputs, num_outputs)

                net += [prop_mid, activation()]
                if params.chemvae_prop_pred_dropout > 0:
                    net += [nn.Dropout(params.chemvae_prop_pred_dropout)]
                if params.chemvae_prop_batchnorm:
                    net += [nn.BatchNorm1d(num_outputs)]

                num_inputs = num_outputs

        net += [nn.Linear(num_inputs, 2)]
        self.net = nn.ModuleList(net)

    def forward(self, x):
        for h in self.net:
            x = h(x)
        means = x[:, 0]
        variances = torch.sigmoid(x[:, 1])
        return means, variances

    def input_shape(self):
        return [self.params.chemvae_prop_pred_num_input_features]


def z_to_smiles(vae, z, decode_attempts=1000, noise_norm=1.0):
    Z = np.tile(z, (decode_attempts, 1))
    Z = vae.perturb_z(Z, noise_norm)
    X = vae.decode(Z)
    smiles = [k for k in vae.hot_to_smiles(X, strip=True) if mu.good_smiles(k)]
    return smiles


def acquire_properties(encoded, vae, decode_attempts=1000, noise=0.1, device='cuda'):
    assert isinstance(encoded, np.ndarray)
    smiles = z_to_smiles(vae, encoded, decode_attempts=decode_attempts, noise_norm=noise)
    batch_size = len(smiles)

    props = torch.zeros([batch_size, 3], device=device)
    props = []
    for i in range(batch_size):
        try:
            scores = utils.all_scores(smiles[i])
            props += [scores]
        except Exception as ex:
            print(ex)
    props = torch.tensor(props, device=device)
    return props


def load_zinc250k(num_to_load=0, score_fn=None, batch_size=128):
    from chemvae_keras import vae_utils
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
    labels = None

    if score_fn is not None:
        labels = np.array([score_fn(k) for k in zinc250k[1]])
        sort_idx = np.argsort(labels)
        zinc250k[0] = [zinc250k[0][i] for i in sort_idx]
        zinc250k[1] = zinc250k[1][sort_idx, :]
        labels = labels[sort_idx]

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
