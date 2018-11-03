#! /usr/bin/env python

from argparse import ArgumentParser
from comet_ml import Experiment
import torch
import pyro.optim
import numpy as np
import pandas as pd
import os
import pickle
from gpu_utils.utils import gpu_init
from bb_opt.src.bayesian_opt import optimize
from bb_opt.src.deep_ensemble import NNEnsemble
from bb_opt.src.bnn import BNN
from bb_opt.src.acquisition_functions import ExpectedReward, UCB, HAF
from bb_opt.src.hsic import mixrq_kernels, mixrbf_kernels
from bb_opt.src.utils import get_path


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

    af = select_af(af_key, args, exp)
    af_greedy = (
        select_af(greedy_af_key, args, exp, greedy=True) if args.greedy_af_key else None
    )

    model = select_model(model_key, device, inputs.shape[1], args, exp)

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


def select_af(af_key: str, args, exp, greedy: bool = False):
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
    elif af_key == "haf":
        af_kwargs = {"target_distr": "MES", "calibrated": False}
        af = HAF(exp=exp, **af_kwargs)
    else:
        assert False, f"AF {af_key} not recognized."

    if greedy:
        af_kwargs = {f"greedy_{key}": value for key, value in af_kwargs.items()}

    exp.log_multiple_params(af_kwargs)
    return af


def select_model(model_key: str, device, n_inputs, args, exp):
    model_kwargs = {"n_inputs": n_inputs, "n_hidden": 100, "device": device}

    if model_key == "de":
        model_kwargs.update(
            {
                "n_models": args.n_ensemble_models,
                "adversarial_epsilon": None,
                "nonlinearity_names": None,
                "extra_random": False,
            }
        )
        model = NNEnsemble.get_model(**model_kwargs)

        lr = 0.01
        model.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        exp.log_parameter("lr", lr)
    elif model_key == "bnn":
        model_kwargs.update(
            {"non_linearity": "ReLU", "prior_mean": 0.0, "prior_std": 1.0}
        )
        model = BNN.get_model(**model_kwargs)

        lr = 0.01
        model.optimizer = pyro.optim.Adam({"lr": lr})
        exp.log_parameter("lr", lr)
    else:
        assert False, f"Unrecognized model key: {model_key}"

    del model_kwargs["device"]
    exp.log_multiple_params(model_kwargs)
    return model


if __name__ == "__main__":
    main()
