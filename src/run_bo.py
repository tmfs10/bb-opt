#! /usr/bin/env python

from argparse import ArgumentParser
import torch
from comet_ml import Experiment
import numpy as np
import pandas as pd
import os
import pickle
from gpu_utils.utils import gpu_init
from bb_opt.src.bayesian_opt import optimize
from bb_opt.src.deep_ensemble import NNEnsemble
from bb_opt.src.bnn import BNN
from bb_opt.src.acquisition_functions import ExpectedReward, UCB
from bb_opt.src.hsic import mixrq_kernels, mixrbf_kernels
from bb_opt.src.utils import get_path

models = {
    # "uniform": get_model_uniform,
    # "nn": get_model_nn,
    "bnn": BNN,
    "de": NNEnsemble,
}

acquisition_functions = {
    # "uniform": acquire_batch_uniform,
    # "nn": acquire_batch_nn_greedy,
    # "ei": acquire_batch_ei,
    # "pdts": acquire_batch_pdts,
    # "hsic_ms": acquire_batch_hsic_mean_std,
    # "hsic_pdts": acquire_batch_hsic_pdts,
    # "mves": acquire_batch_mves,
    # "es": acquire_batch_es,
    # "pi": acquire_batch_pi,
    "er": ExpectedReward,
    "ucb": UCB,
}


def main():
    parser = ArgumentParser()
    parser.add_argument("model_key", help="One of {bnn, de}")
    parser.add_argument("af_key", help="One of {er}")
    parser.add_argument("project", help="One of {malaria, dna_binding}")
    parser.add_argument(
        "--dataset",
        default="",
        help="Needed for projects w/ multiple datasets e.g. 'crx_ref_r1' for for dna_binding",
    )
    parser.add_argument(
        "-s",
        "--save_key",
        help="Key used in the saved dictionary of results [default = `model_key` + `af_key`]",
    )
    parser.add_argument("-d", "--device_num", type=int, default=-1)
    parser.add_argument("-b", "--batch_size", type=int, default=100)
    parser.add_argument(
        "-n",
        "--n_epochs",
        type=int,
        default=-1,
        help="-1 means train until early stopping occurs; this must be supported by the model function used.",
    )
    parser.add_argument(
        "-k",
        "--kernel",
        default="rq",
        help="Kernel for HSIC; one of {rq, rbf} [default = rq]",
    )
    parser.add_argument("-c", "--hsic_coeff", type=float, default=150.0)
    parser.add_argument("--hsic_metric", default="mu / sigma - hsic")
    parser.add_argument("--preds_multiplier", type=float, default=2.5)
    parser.add_argument("--pdts_multiplier", type=float, default=2.0)
    parser.add_argument(
        "-nb",
        "--n_batches",
        type=int,
        default=-1,
        help="Run until this many batches have been acquired. -1 (default) means to train until the entire dataset has been acquired.",
    )
    parser.add_argument(
        "-r",
        "--retrain",
        type=int,
        default=1,
        help="Retrain after this many batches acquired.",
    )
    parser.add_argument(
        "-p",
        "--partial_epochs",
        type=int,
        default=10,
        help="Train for this many epochs on new data when not full retraining.",
    )
    parser.add_argument(
        "-mi", "--mi_estimator", default="HSIC", help="One of {HSIC, LNC}"
    )
    parser.add_argument("-nes", "--n_points_es", type=int, default=1)
    parser.add_argument("-nde", "--n_ensemble_models", type=int, default=5)
    parser.add_argument(
        "-gk",
        "--greedy_af_key",
        help="AF key to use when being exploitative on the final batch [default = af_key].",
    )
    parser.add_argument("-cm", "--combine_max_dists", action="store_true")
    parser.add_argument("-cal", "--calibrated", action="store_true")
    parser.add_argument(
        "-beta",
        "--beta",
        type=float,
        default=0.0,
        help="Beta value for UCB [default = 0.0]",
    )
    args = parser.parse_args()

    model_key = args.model_key
    af_key = args.af_key
    greedy_af_key = args.greedy_af_key or af_key
    save_key = args.save_key or "_".join((model_key, af_key))
    mi_estimator = args.mi_estimator

    device_num = None if args.device_num == -1 else args.device_num
    device = gpu_init(device_num, ml_library="torch", verbose=True)

    root = get_path(__file__, "..", "..")
    data_dir = get_path(root, "data", args.project, args.dataset)
    inputs = np.load(get_path(data_dir, "inputs.npy"))
    labels = np.load(get_path(data_dir, "labels.npy"))

    top_k_percent = 1

    with open(f"{os.environ['HOME']}/.comet_key") as f:
        comet_key = f.read().strip()

    exp = Experiment(
        comet_key,
        workspace="bayesian-optimization",
        project_name=args.project,
        auto_param_logging=False,
        parse_args=False,
    )

    exp.log_multiple_params(
        {
            "model_key": model_key,
            "af_key": af_key,
            "greedy_af_key": greedy_af_key,
            "save_key": save_key,
            "dataset": args.dataset,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "n_batches": args.n_batches,
            "retrain": args.retrain,
            "partial_epochs": args.partial_epochs,
        }
    )

    af_greedy = None
    if af_key in ("er", "ucb"):
        if args.calibrated:
            af_kwargs = {"calibrated": True, "gaussian_approx": True}
        else:
            af_kwargs = {}

        if af_key == "er":
            af = ExpectedReward(exp=exp, **af_kwargs)
        else:
            af_kwargs["beta"] = args.beta
            af = UCB(exp=exp, **af_kwargs)
        exp.log_multiple_params(af_kwargs)

    if model_key == "de":
        model_kwargs = {
            "n_inputs": inputs.shape[1],
            "n_models": args.n_ensemble_models,
            "n_hidden": 100,
            "adversarial_epsilon": None,
            "device": device,
            "nonlinearity_names": None,
            "extra_random": False,
        }
        model = NNEnsemble.get_model(**model_kwargs)

        lr = 0.01
        model.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        exp.log_parameter("lr", lr)

        del model_kwargs["device"]

    exp.log_multiple_params(model_kwargs)

    optimize(
        model,
        af,
        inputs,
        labels,
        top_k_percent,
        args.batch_size,
        args.n_epochs,
        device,
        verbose=True,
        exp=exp,
        n_batches=args.n_batches,
        retrain_every=args.retrain,
        partial_train_epochs=args.partial_epochs,
        save_key=save_key,
        acquisition_func_greedy=af_greedy,
    )


if __name__ == "__main__":
    main()
