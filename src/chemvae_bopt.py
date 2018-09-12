
import torch
import torch.nn as nn
import matlab.engine
import gp_acquisition as ack
import numpy as np

from chemvae_keras import models
from chemvae_keras import vae_utils
from chemvae_keras import mol_utils as mu

def chemvae_gpopt_pes(is_cuda, zinc250k, directory, n_init_samples, n_opt_samples, obj):
    assert n_init_samples < len(zinc250k)
    vae = vae_utils.VAEUtils(directory=directory)

    zinc250k.sort(key=lambda k: obj(k[1]))
    smiles = []
    for smiles_string, props in zinc250k:
        smiles += [vae.smiles_to_hot(mu.canon_smiles(smiles_string), canonize_smiles=True)]

    x_samples = np.asarray([vae.encode[k] for k in smiles[:n_init_samples]], dtype=np.float32)
    predictor_model = lambda x : obj(vae.predict_prop_Z(x)[0])
    nvars = x_samples.shape[1]
    x_min = np.ones(nvars)*-1
    x_max = np.ones(nvars)

    eng = matlab.engine.start_matlab()

    x_samples, y_samples, l, sigma, sigma0 = ack.init_pes(is_cuda, eng, predictor_model, x_samples, n_opt_samples, x_min, x_max)

    for i in range(num_batches):
        x_samples, y_samples, guesses, l, sigma, sigma0 = ack.pes(eng, obj, n_opt_samples, n_features, x_min, x_max, x_samples, guesses, y_samples, l, sigma, sigma0)

    eng.quit()

    return np.asarray(x_samples), np.asarray(y_samples), np.asarray(guesses)
