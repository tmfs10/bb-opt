#! /usr/bin/env python

from argparse import ArgumentParser
import torch
from comet_ml import Experiment
import numpy as np
import pandas as pd
import os
import pickle
from bb_opt.src.bayesian_opt import (
    optimize,
    get_model_uniform,
    get_model_nn,
    get_model_bnn,
    acquire_batch_uniform,
    acquire_batch_uniform,
    acquire_batch_nn_greedy,
    acquire_batch_bnn_greedy,
    acquire_batch_pdts,
    acquire_batch_hsic,
    train_model_uniform,
    train_model_nn,
    train_model_bnn,
)

models = {
    "uniform": get_model_uniform,
    "nn": get_model_nn,
    "bnn": get_model_bnn,
    "pdts": get_model_bnn,
    "hsic": get_model_bnn,
}

acquisition_functions = {
    "uniform": acquire_batch_uniform,
    "nn": acquire_batch_nn_greedy,
    "bnn": acquire_batch_bnn_greedy,
    "pdts": acquire_batch_pdts,
    "hsic": acquire_batch_hsic,
}

train_functions = {
    "uniform": train_model_uniform,
    "nn": train_model_nn,
    "bnn": train_model_bnn,
    "pdts": train_model_bnn,
    "hsic": train_model_bnn,
}


def _get_path(*path_segments):
    return os.path.realpath(os.path.join(*path_segments))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_key", help="One of [uniform, nn, bnn, pdts, hsic]")
    parser.add_argument(
        "-s",
        "--save_key",
        help="Key used in the saved dictionary of results [default = `model_key`]",
    )
    parser.add_argument("-d", "--device_num", type=int, default=0)
    parser.add_argument("-b", "--batch_size", type=int, default=200)
    parser.add_argument(
        "-n",
        "--n_epochs",
        type=int,
        default=-1,
        help="-1 means train until early stopping occurs; this must be supported by the model function used.",
    )
    args = parser.parse_args()

    model_key = args.model_key
    save_key = args.save_key or model_key
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root = os.path.realpath(os.path.join(__file__, "../.."))
    fingerprints = np.load(_get_path(root, "data/fingerprints.npy"))
    ec50 = pd.read_csv(_get_path(root, "data/malaria.csv")).ec50.values

    n_repeats = 1
    top_k_percent = 1

    with open(f"{os.environ['HOME']}/.comet_key") as f:
        comet_key = f.read().strip()

    exp = Experiment(comet_key, project_name="bb_opt", auto_param_logging=False)
    exp.log_multiple_params(
        {
            "model_key": model_key,
            "save_key": save_key,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
        }
    )

    if model_key == "hsic":
        # TODO: these should be command line args too;
        # maybe `optimize` should accept kwargs for ac func and train func?
        exp.log_multiple_params(
            {
                "hsic_coeff": 150,
                "n_preds": 250,
                "metric": "mu / sigma - hsic",
                "kernel": "linear",
            }
        )

    mean, std = optimize(
        models[model_key],
        acquisition_functions[model_key],
        train_functions[model_key],
        fingerprints,
        ec50,
        top_k_percent,
        n_repeats,
        args.batch_size,
        args.n_epochs,
        device,
        verbose=True,
        exp=exp,
    )
    fraction_best = {save_key: (mean, std)}

    rand = np.random.randint(1000000)
    fname = _get_path(root, f"figures/plot_data/fraction_best_{model_key}_{rand}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(fraction_best, f)
