import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors
from typing import Callable, Tuple, Sequence
from bb_opt.src.utils import jointplot


def get_confidence_interval_gaussian(
    mean: np.ndarray, std: np.ndarray, conf_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the lower and upper bounds of a credible region containing `conf_level` probability mass.

    The bounds will be centered around the corresponding mean. Formula according to Wikipedia.

    :param mean: size n_inputs
    :param std: size n_inputs
    :param conf_level:
    :returns: lower_bound, upper_bound (both of size n_inputs)
    """
    std_normal_quantile = 1 - (1 - conf_level) / 2
    n_std = (
        torch.sqrt(torch.tensor(2.))
        * torch.erfinv(torch.tensor(2 * std_normal_quantile - 1))
    ).item()
    lower_bound = mean - n_std * std
    upper_bound = mean + n_std * std
    return lower_bound, upper_bound


def get_confidence_interval_ecdf(
    preds: np.ndarray, conf_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param preds: size n_samples x n_inputs
    :param conf_leve:
    :returns: lower_bound, upper_bound (both of size n_inputs)
    """
    lower_quantile, upper_quantile = get_confidence_interval_cdf(conf_level)
    lower_bound = np.quantile(preds, lower_quantile, axis=0, interpolation="lower")
    upper_bound = np.quantile(preds, upper_quantile, axis=0, interpolation="higher")
    assert (
        ((lower_bound <= preds) & (preds <= upper_bound)).mean(axis=0) >= conf_level
    ).all()
    return lower_bound, upper_bound


def get_confidence_interval_cdf(conf_level: float = 0.95) -> Tuple[float, float]:
    """
    Compute the lower and upper bounds on the CDF for `conf_level`.

    The bounds are such that (upper_bound - lower_bound) == conf_level and they're
    centered around 0.5.
    :returns: lower_bound, upper_bound
    """
    lower_bound = (1 - conf_level) / 2
    upper_bound = 1 - lower_bound
    assert np.isclose(upper_bound - lower_bound, conf_level)
    return lower_bound, upper_bound


def get_gaussian_cdf_probs(pred_means, pred_stds, labels):
    if not isinstance(pred_means, torch.Tensor):
        pred_means = torch.tensor(pred_means)

    if not isinstance(pred_stds, torch.Tensor):
        pred_stds = torch.tensor(pred_stds)

    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    sqrt2 = torch.sqrt(pred_means.new_tensor(2.0))
    cdf_probs = 0.5 * (1 + torch.erf((labels - pred_means) / (pred_stds * sqrt2)))
    return cdf_probs


def get_ecdf_probs(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute the probability of each label under the eCDF for the corresponding input.

    :param preds: n_samples x n_inputs
    :param labels: n_inputs
    :returns: n_inputs
    """
    ecdf_probs = (preds <= labels).mean(0)
    return ecdf_probs


def calibrate_cdf_probs(cdf_probs, return_regressor: bool = False):
    """
    Follows the method from https://arxiv.org/pdf/1807.00263.pdf.
    """
    empirical_cdf_probs = []
    for cdf_prob in cdf_probs:
        empirical_cdf_probs.append((cdf_probs <= cdf_prob).mean())
    empirical_cdf_probs = np.array(empirical_cdf_probs)

    calibration_regressor = IsotonicRegression(y_min=0, y_max=1)
    calibrated_cdf_probs = calibration_regressor.fit_transform(
        cdf_probs, empirical_cdf_probs
    )
    if return_regressor:
        return calibrated_cdf_probs, calibration_regressor
    return calibrated_cdf_probs


def plot_calibration(
    preds,
    labels,
    calibrate: bool = True,
    plot_cdf_pdf: bool = True,
    gaussian_approx: bool = False,
):
    conf_levels = np.arange(0.01, 1.01, 0.01)

    if gaussian_approx:
        pred_means = preds.mean(0)
        pred_stds = preds.std(0)

    observed_conf_levels = []
    for conf_level in conf_levels:
        if gaussian_approx:
            lower_bound, upper_bound = get_confidence_interval_gaussian(
                pred_means, pred_stds, conf_level=conf_level
            )
        else:
            lower_bound, upper_bound = get_confidence_interval_ecdf(preds, conf_level)

        observed_conf_levels.append(
            ((lower_bound <= labels) & (labels <= upper_bound)).mean()
        )

    if calibrate or plot_cdf_pdf:
        if gaussian_approx:
            cdf_probs = get_gaussian_cdf_probs(pred_means, pred_stds, labels).numpy()
        else:
            cdf_probs = get_ecdf_probs(preds, labels)

    if calibrate:
        calibrated_cdf_probs = calibrate_cdf_probs(cdf_probs)

        observed_conf_levels_calibrated = []
        for conf_level in conf_levels:
            lower_bound, upper_bound = get_confidence_interval_cdf(
                conf_level=conf_level
            )
            observed_conf_levels_calibrated.append(
                (
                    (lower_bound <= calibrated_cdf_probs)
                    & (calibrated_cdf_probs <= upper_bound)
                ).mean()
            )

    n_cols = 1 + plot_cdf_pdf
    fig, axes = plt.subplots(ncols=n_cols, figsize=(6 * n_cols, 6), squeeze=False)

    ax = axes[0, 0]
    ax.scatter(conf_levels, observed_conf_levels, s=5)
    ax.plot(conf_levels, observed_conf_levels, label="Uncalibrated")

    if calibrate:
        ax.scatter(conf_levels, observed_conf_levels_calibrated, s=5)
        ax.plot(conf_levels, observed_conf_levels_calibrated, label="Calibrated")

    ax.set_xlabel("Expected Confidence Level")
    ax.set_ylabel("Observed Confidence Level")
    ax.set_title("Calibration Plot")
    ax.legend()

    if not plot_cdf_pdf:
        return

    ax = axes[0, 1]
    ax.set_title("Distribution of CDF Probs")
    sns.distplot(cdf_probs, label="Uncalibrated", ax=ax)

    if calibrate:
        sns.distplot(calibrated_cdf_probs, label="Calibrated", ax=ax)

    ax.set_xlabel("CDF Prob Value")
    ax.legend()


def plot_means_stds(pred_means, pred_stds):
    fig, axes = plt.subplots(ncols=2, figsize=(8, 3))

    ax = axes[0]
    sns.distplot(pred_means, ax=ax)
    ax.set_xlabel("Mean")
    ax.set_title("Distribution of Predicted Mean")

    ax = axes[1]
    sns.distplot(pred_stds, ax=ax)
    ax.set_xlabel("Std")
    ax.set_title("Distribution of Predicted Standard Deviation")


def plot_std_vs_mse(pred_means: np.ndarray, pred_stds: np.ndarray, labels: np.ndarray):
    mses = (pred_means - labels) ** 2
    jointplot(pred_stds, mses, ("Pred Std", "MSE"), title="Pred. Std. Dev. vs MSE")


def get_hidden_reprs_bnn(
    guide: Callable, inputs: np.ndarray, n_samples: int = 1000
) -> np.ndarray:
    inputs = next(guide().parameters()).new_tensor(inputs)
    hidden_reprs = []

    for _ in range(n_samples):
        nn_sample = guide()
        layers = list(nn_sample.children())

        with torch.no_grad():
            hidden_reprs.append(layers[1](layers[0](inputs)).cpu().numpy())

    hidden_reprs = np.mean(hidden_reprs, 0)
    return hidden_reprs


def plot_nn_dist_mse_std(
    train_inputs, inputs, pred_means, pred_stds, labels, dist_metric="l2"
):
    train_neighbors = NearestNeighbors(
        n_neighbors=1, algorithm="ball_tree", metric=dist_metric
    ).fit(train_inputs)

    nn_dists = train_neighbors.kneighbors(inputs)[0].squeeze()

    mses = (pred_means - labels) ** 2

    if (nn_dists != 0).any():
        jointplot(
            nn_dists, mses, ("Nearest Neighbor Dist.", "MSE"), title="NN Dist vs MSE"
        )
        jointplot(
            nn_dists,
            pred_stds,
            ("Nearest Neighbor Dist.", "Pred. Std. Dev."),
            title="NN Dist vs Std Dev",
        )


def plot_pdfs(
    preds: np.ndarray,
    labels: np.ndarray,
    n_pdfs: int = 10,
    gaussian_approx: bool = False,
):
    """
    :param preds: size n_samples x n_inputs
    """
    n_cols = 5
    n_rows = int(np.ceil(n_pdfs / n_cols))
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows)
    )
    axes = [ax for ax_row in axes for ax in ax_row]  # flatten

    for i, idx in enumerate(np.random.choice(range(len(preds)), size=n_pdfs)):
        ax = axes[i]
        sns.distplot(preds[:, i], label="True", ax=ax)

        if gaussian_approx:
            sns.distplot(
                np.random.normal(preds[:, i].mean(), preds[:, i].std(), size=1000),
                label="Normal Approx.",
                ax=ax,
            )

        ax.axvline(labels[i], color="red", label="Obs")

        if i == 0:
            ax.legend(
                loc=3,
                bbox_to_anchor=(0., 1.02, 1., .102),
                ncol=3 if gaussian_approx else 2,
            )


def plot_preds_with_conf_intervals(
    preds: np.ndarray,
    labels,
    conf_level: float = 0.95,
    gaussian_approx: bool = False,
    calibrate: bool = True,
):
    """
    Plot the mean predictions for each input with confidence intervals and the true labels.

    :param preds: size n_samples x n_inputs
    :param labels: size n_inputs
    :param conf_level:
    :param gaussian_approx: whether to approximate the distribution for each input as a Gaussian
    """
    pred_means = preds.mean(0)

    if gaussian_approx:
        pred_stds = preds.std(0)

        if calibrate:
            cdf_probs = get_gaussian_cdf_probs(pred_means, pred_stds, labels).numpy()
            _, calibration_regressor = calibrate_cdf_probs(
                cdf_probs, return_regressor=True
            )
            calibrated_conf_level = get_calibrated_conf_level(
                calibration_regressor, conf_level
            )
            calibrated_lower_bound, calibrated_upper_bound = get_confidence_interval_gaussian(
                pred_means, pred_stds, conf_level=calibrated_conf_level
            )

        lower_bound, upper_bound = get_confidence_interval_gaussian(
            pred_means, pred_stds, conf_level=conf_level
        )
    else:
        if calibrate:
            cdf_probs = get_ecdf_probs(preds, labels)
            _, calibration_regressor = calibrate_cdf_probs(
                cdf_probs, return_regressor=True
            )
            calibrated_conf_level = get_calibrated_conf_level(
                calibration_regressor, conf_level
            )
            calibrated_lower_bound, calibrated_upper_bound = get_confidence_interval_ecdf(
                preds, calibrated_conf_level
            )

        lower_bound, upper_bound = get_confidence_interval_ecdf(preds, conf_level)

    # sorting makes the plot much nicer + easier to interpret
    sorted_idx = np.argsort(pred_means)
    lower_bound = lower_bound[sorted_idx]
    upper_bound = upper_bound[sorted_idx]
    pred_means = pred_means[sorted_idx]
    labels = labels[sorted_idx]

    if calibrate:
        calibrated_lower_bound = calibrated_lower_bound[sorted_idx]
        calibrated_upper_bound = calibrated_upper_bound[sorted_idx]

    fig = plt.figure(figsize=(10, 5))
    plt.title("Predictions + Confidence Intervals")
    plt.scatter(range(len(labels)), labels, label="True", s=1, c="red")
    plt.scatter(range(len(pred_means)), pred_means, s=1, label="Pred Mean", c="blue")
    # plt.vlines(
    #     range(len(pred_means)),
    #     lower_bound,
    #     upper_bound,
    #     alpha=0.1,
    #     color="blue",
    #     label=f"{100 * conf_level:g}% Conf",
    # )

    plt.scatter(
        range(len(pred_means)),
        lower_bound,
        s=1,
        alpha=0.3,
        color="blue",
        label=f"{100 * conf_level:g}% Conf",
    )

    plt.scatter(range(len(pred_means)), upper_bound, s=1, alpha=0.3, color="blue")

    if calibrate:
        plt.scatter(
            range(len(pred_means)),
            calibrated_lower_bound,
            s=1,
            alpha=0.3,
            color="orange",
            label=f"Calibrated {100 * conf_level:g}% Conf",
        )

        plt.scatter(
            range(len(pred_means)),
            calibrated_upper_bound,
            s=1,
            alpha=0.3,
            color="orange",
        )

    plt.xlabel("Sorted Input Index")
    plt.ylabel("Value")
    plt.legend()


def get_uncalibrated_cdf_prob_for_cdf_prob(
    calibration_regressor, target_cdf_prob: float
) -> float:
    """
    Find the uncalibrated CDF prob which gives `target_cdf_prob` when calibrated.
    """
    min_x = calibration_regressor.X_min_
    max_x = calibration_regressor.X_max_

    while True:
        mid_x = min_x + (max_x - min_x) / 2
        current_cdf_prob = calibration_regressor.transform([mid_x])[0]

        if abs(current_cdf_prob - target_cdf_prob) <= 1e-6 or max_x - min_x <= 1e-6:
            break
        elif current_cdf_prob > target_cdf_prob:
            max_x = mid_x
        elif current_cdf_prob < target_cdf_prob:
            min_x = mid_x
    return mid_x


def get_calibrated_conf_level(calibration_regressor, conf_level: float) -> float:
    lower_cdf_prob, upper_cdf_prob = get_confidence_interval_cdf(conf_level)
    lower_cdf_prob = get_uncalibrated_cdf_prob_for_cdf_prob(
        calibration_regressor, lower_cdf_prob
    )
    upper_cdf_prob = get_uncalibrated_cdf_prob_for_cdf_prob(
        calibration_regressor, upper_cdf_prob
    )
    calibrated_conf_level = upper_cdf_prob - lower_cdf_prob
    return calibrated_conf_level


def diagnostic_plots(
    inputs,
    train_inputs,
    preds,
    labels,
    conf_level,
    guide=None,
    gaussian_approx: bool = False,
    plot_list: Sequence[str] = (),
):
    pred_means = preds.mean(0)
    pred_stds = preds.std(0)

    plot_preds_with_conf_intervals(preds, labels, conf_level=conf_level)
    plot_pdfs(preds, labels, gaussian_approx=gaussian_approx)
    plot_means_stds(pred_means, pred_stds)
    plot_calibration(
        preds,
        labels,
        calibrate=True,
        plot_cdf_pdf=True,
        gaussian_approx=gaussian_approx,
    )
    plot_std_vs_mse(pred_means, pred_stds, labels)
    plot_nn_dist_mse_std(train_inputs, inputs, pred_means, pred_stds, labels, "hamming")

    if guide:
        train_hidden_reprs = get_hidden_reprs_bnn(guide, train_inputs)
        hidden_reprs = get_hidden_reprs_bnn(guide, inputs)
        plot_nn_dist_mse_std(
            train_hidden_reprs, hidden_reprs, pred_means, pred_stds, labels, "l2"
        )
