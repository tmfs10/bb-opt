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
    acquire_batch_nn_greedy,
    acquire_batch_bnn_greedy,
    acquire_batch_pdts,
    acquire_batch_hsic_mean_std,
    acquire_batch_hsic_pdts,
    acquire_batch_mves,
    acquire_batch_es,
    acquire_batch_pi,
    train_model_uniform,
    train_model_nn,
    train_model_bnn,
)
from bb_opt.src.hsic import dimwise_mixrq_kernels, dimwise_mixrbf_kernels
from bb_opt.src.utils import get_path

models = {
    "uniform": get_model_uniform,
    "nn": get_model_nn,
    "bnn": get_model_bnn,
    "pdts": get_model_bnn,
    "hsic_ms": get_model_bnn,
    "hsic_pdts": get_model_bnn,
    "mves": get_model_bnn,
    "es": get_model_bnn,
    "pi": get_model_bnn,
}

acquisition_functions = {
    "uniform": acquire_batch_uniform,
    "nn": acquire_batch_nn_greedy,
    "bnn": acquire_batch_bnn_greedy,
    "pdts": acquire_batch_pdts,
    "hsic_ms": acquire_batch_hsic_mean_std,
    "hsic_pdts": acquire_batch_hsic_pdts,
    "mves": acquire_batch_mves,
    "es": acquire_batch_es,
    "pi": acquire_batch_pi,
}

train_functions = {
    "uniform": train_model_uniform,
    "nn": train_model_nn,
    "bnn": train_model_bnn,
    "pdts": train_model_bnn,
    "hsic_ms": train_model_bnn,
    "hsic_pdts": train_model_bnn,
    "mves": train_model_bnn,
    "es": train_model_bnn,
    "pi": train_model_bnn,
}


def main():
    parser = ArgumentParser()
    parser.add_argument("model_key", help="One of {uniform, nn, bnn, pdts, hsic}")
    parser.add_argument("project", help="One of {malaria, dna_binding}")
    parser.add_argument(
        "--dataset",
        default="",
        help="Needed for projects w/ multiple datasets e.g. 'crx_ref_r1' for for dna_binding",
    )
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
    parser.add_argument("-k", "--kernel", default="rq", help="rq (default) or rbf")
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
    args = parser.parse_args()

    model_key = args.model_key
    save_key = args.save_key or model_key
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            "save_key": save_key,
            "dataset": args.dataset,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "n_batches": args.n_batches,
            "n_hidden": 100,
        }
    )

    acquisition_args = {}
    if model_key in ["es", "mves", "hsic_ms", "hsic_pdts", "pdts"]:
        kernels = {"rq": dimwise_mixrq_kernels, "rbf": dimwise_mixrbf_kernels}
        acquisition_args["kernel"] = args.kernel

        if model_key not in ("es", "mves"):
            # just because, for now at least, we report HSIC during PDTS acquisition,
            # even though the HSIC doesn't affect the result
            acquisition_args[
                "preds_multiplier"
            ] = args.preds_multiplier  # n_preds = preds_mult * batch_size

    if model_key == "hsic_ms":
        acquisition_args["hsic_coeff"] = args.hsic_coeff
        acquisition_args["metric"] = args.hsic_metric

    if model_key == "hsic_pdts":
        acquisition_args["pdts_multiplier"] = args.pdts_multiplier

    exp.log_multiple_params(acquisition_args)

    if "kernel" in acquisition_args:
        acquisition_args["kernel"] = kernels[acquisition_args["kernel"]]

    optimize(
        models[model_key],
        acquisition_functions[model_key],
        train_functions[model_key],
        inputs,
        labels,
        top_k_percent,
        args.batch_size,
        args.n_epochs,
        device,
        verbose=True,
        exp=exp,
        acquisition_args=acquisition_args,
        n_batches=args.n_batches,
    )


if __name__ == "__main__":
    main()
