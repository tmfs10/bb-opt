import numpy as np
from scipy.special import erfinv
from sklearn.isotonic import IsotonicRegression
from typing import Tuple, Optional


class CDF:
    """
    WARNING - there still seem to be some bugs with this class!
    """

    def __init__(
        self,
        preds: Optional[np.ndarray] = None,
        calibration_regressor: Optional = None,
        gaussian_approx: bool = False,
        pred_means_stds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        calibration_method: str = "cdf",
    ):
        """
        :param preds: samples to use in construction the (e)CDFs; there will be one CDF
          per input (shape n_samples x n_inputs)
        :param calibration_regressor: a trained regressor (e.g. IsotonicRegression) that takes
          in uncalibrated CDF probabilities and returns calibrated ones. This is most often used
          if you've trained a regressor on a `CDF` for the training data and now want to use it to
          calibrate a `CDF` on the test data
        :param gaussian_approx: whether to approximate the CDF as a Gaussian. If not,
          the emprical CDF (eCDF) is used instead
        :param pred_means_stds: only used if `gaussian_approx == True`; these are the means
          and standard deviations of each of the Gaussians. If not given, these are generated
          by taking computing the mean + std. dev. across the samples for each input (shapes: n_inputs)
        :param calibration_method: one of {cdf, conf}
          cdf uses the method from https://arxiv.org/pdf/1807.00263.pdf (inputs/outputs are CDF probabilities)
          conf regresses on the calibration curve itself (inputs/outputs are confidence levels)
          conf is only supported for calibrating confidence intervals, not sampling or computing CDF probabilties
        """
        self.gaussian_approx = gaussian_approx
        self.n_samples, self.n_inputs = (
            preds.shape if preds is not None else (None, None)
        )

        if gaussian_approx:
            if pred_means_stds:
                self.pred_means, self.pred_stds = pred_means_stds
                self.n_inputs = len(self.pred_means)
            else:
                self.pred_means = preds.mean(0)
                self.pred_stds = preds.std(0)
        else:
            assert (
                preds is not None
            ), "Must pass `preds` unless using `gaussian_approx`."
            self.preds = np.sort(preds, axis=0)

        self.calibration_method = calibration_method
        self.calibration_regressor = calibration_regressor
        self._expected_conf = self._observed_conf = None

    def get_cdf_probs(self, x: np.ndarray, calibrated: bool = False) -> np.ndarray:
        """
        Get the probability of each element of `x` under the corresponding CDF.

        :param x: (shape: n_inputs or n_inputs x n_queries)
        :returns: (shape: n_inputs x n_queries)
        """
        x = x if x.ndim == 2 else x[:, None]  # add n_queries dimension

        if self.gaussian_approx:
            cdf_probs = self._get_gaussian_cdf_probs(x)
        else:
            cdf_probs = self._get_ecdf_probs(x)

        if not calibrated:
            return cdf_probs

        assert self.calibration_method == "cdf"

        return self.calibrate_cdf_probs(cdf_probs)

    def get_calibration_curve(
        self, labels: np.ndarray, calibrated: bool = False, n_conf_levels: int = 101
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :returns: expected_conf, observed_conf
        """
        expected_conf = np.linspace(0, 1, n_conf_levels)

        observed_conf = []
        for conf_level in expected_conf:
            lower, upper = self.get_confidence_intervals(conf_level, calibrated)
            observed_conf.append(((lower <= labels) & (labels <= upper)).mean())
        observed_conf = np.array(observed_conf)

        return expected_conf, observed_conf

    def train_cdf_calibrator(self, labels: np.ndarray) -> np.ndarray:
        """
        Trains a regressor to calibrate the CDF probabilities based on emprical results `x`.
        """
        calibration_regressor = IsotonicRegression(
            y_min=0, y_max=1, out_of_bounds="clip"
        )
        self.calibration_regressor = calibration_regressor

        if self.calibration_method == "cdf":
            cdf_probs = self.get_cdf_probs(labels).squeeze()
            empirical_cdf_probs = []
            for cdf_prob in cdf_probs:
                empirical_cdf_probs.append((cdf_probs <= cdf_prob).mean())
            empirical_cdf_probs = np.array(empirical_cdf_probs)

            calibrated_cdf_probs = calibration_regressor.fit_transform(
                cdf_probs, empirical_cdf_probs
            )
            return calibrated_cdf_probs
        else:
            assert self.calibration_method == "conf"

            expected_conf, observed_conf = self.get_calibration_curve(labels)

            calibrated_confs = calibration_regressor.fit_transform(
                observed_conf, expected_conf
            )
            return calibrated_confs

    def calibrate_cdf_probs(self, cdf_probs: np.ndarray) -> np.ndarray:
        """
        :param cdf_probs: (shape: any; it will be `ravel`ed before being calibrated)
        :returns: (shape: `cdf_probs.shape`)
        """
        assert (
            self.calibration_regressor is not None
        ), "A calibrator must be trained by `train_cdf_calibrator` first."
        calibrated = self.calibration_regressor.transform(cdf_probs.ravel())
        return calibrated.reshape(cdf_probs.shape)

    def get_confidence_intervals(
        self, conf_level: float = 0.95, calibrated: bool = False
    ):
        """
        Compute the lower and upper bounds of a credible region containing `conf_level` probability mass.
        """
        if self.gaussian_approx:
            assert not calibrated or self.calibration_method == "cdf"
            return self._get_confidence_interval_gaussian(conf_level, calibrated)
        return self._get_confidence_interval_ecdf(conf_level, calibrated)

    def sample(self, n_samples: int = 1, calibrated: bool = False) -> np.ndarray:
        """
        Sample from the CDF using inversion sampling.

        :param n_samples: number of samples to draw from each CDF.
        :returns: array with `n_samples` samples from each CDF (shape n_inputs x n_samples)
        """
        assert not calibrated or self.calibration_method == "cdf"
        if self.gaussian_approx:
            return self._sample_gaussian(n_samples, calibrated)
        return self._sample_ecdf(n_samples, calibrated)

    def _get_gaussian_cdf_probs(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the probability of each element of x under the Gaussian for the corresponding input.

        :param x: (shape: n_inputs x n_queries)
        :returns: (shape: n_inputs x n_queries)
        """
        scaled_std = self.pred_stds[:, None] * np.sqrt(2.0)
        cdf_probs = (x - self.pred_means[:, None]) / scaled_std
        cdf_probs = 0.5 * (1 + np.erf(cdf_probs))
        return cdf_probs

    def _get_ecdf_probs(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the probability of each element of x under the eCDF for the corresponding input.

        :param x: (shape: n_inputs x n_queries)
        :returns: (shape: n_inputs x n_queries)
        """
        ecdf_idx = np.array(
            [np.searchsorted(self.preds[:, i], x[i, :]) for i in range(self.n_inputs)]
        )
        ecdf_probs = ecdf_idx / self.n_samples
        return ecdf_probs

    def _sample_gaussian(
        self, n_samples: int = 1, calibrated: bool = False
    ) -> np.ndarray:
        assert (
            not calibrated
        ), "Sampling from a calibrated Gaussian is not yet supported."
        uniform_samples = np.random.uniform(size=(self.n_inputs, n_samples))

        n_std = 10
        min_x = (
            np.tile(self.pred_means, (n_samples, 1)).T - n_std * self.pred_stds[:, None]
        )
        max_x = (
            np.tile(self.pred_means, (n_samples, 1)).T + n_std * self.pred_stds[:, None]
        )

        while not np.allclose(min_x, max_x):
            mid_x = (max_x - min_x) / 2 + min_x
            cdf_probs = self.get_cdf_probs(mid_x)

            max_x[cdf_probs > uniform_samples] = mid_x[cdf_probs > uniform_samples]
            min_x[cdf_probs < uniform_samples] = mid_x[cdf_probs < uniform_samples]

        assert np.isclose(((cdf_probs - uniform_samples) ** 2).mean(), 0)
        return min_x

    def _sample_ecdf(self, n_samples: int = 1, calibrated: bool = False) -> np.ndarray:
        uniform_samples = np.random.uniform(size=(self.n_inputs, n_samples))

        if calibrated:
            cdf_probs = self._get_cdf_probs_of_preds(calibrated, include_zero=True)
            max_idx_le_uniform_sample = np.searchsorted(cdf_probs, uniform_samples)
            max_idx_le_uniform_sample[max_idx_le_uniform_sample > 0] -= 1
        else:
            max_idx_le_uniform_sample = (uniform_samples * self.n_samples).astype(int)

        return self.preds[max_idx_le_uniform_sample.T, range(self.n_inputs)].T

    def _get_cdf_probs_of_preds(
        self, calibrated: bool = False, include_zero: bool = False
    ) -> np.ndarray:
        if include_zero:
            cdf_probs = np.arange(self.n_samples) / self.n_samples
        else:
            cdf_probs = np.arange(1, self.n_samples + 1) / self.n_samples

        if calibrated:
            cdf_probs = self.calibrate_cdf_probs(cdf_probs)
        return cdf_probs

    def _get_confidence_interval_gaussian(
        self, conf_level: float = 0.95, calibrated: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The bounds will be centered around the corresponding mean. Formula according to Wikipedia.
        """
        assert not calibrated, "Calibration isn't supported for Gaussian yet."
        std_normal_quantile = 1 - (1 - conf_level) / 2
        n_std = np.sqrt(2) * erfinv(2 * std_normal_quantile - 1)
        lower_bound = self.pred_means - n_std * self.pred_stds
        upper_bound = self.pred_means + n_std * self.pred_stds
        return lower_bound, upper_bound

    def _get_confidence_interval_ecdf(
        self, conf_level: float = 0.95, calibrated: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param conf_level:
        :returns: lower_bound, upper_bound (both of size n_inputs)
        """
        assert (
            0 <= conf_level <= 1
        ), f"conf_level was {conf_level} but must be in [0, 1]."

        if calibrated and self.calibration_method == "conf":
            conf_level = self.calibration_regressor.transform([conf_level])[0]

        lower_quantile, upper_quantile = self._get_confidence_interval_cdf(conf_level)

        if calibrated and self.calibration_method == "cdf":
            cdf_probs = self._get_cdf_probs_of_preds(calibrated)

            lower_idx = np.searchsorted(cdf_probs, lower_quantile)
            # only don't need this if they're equal
            if cdf_probs[lower_idx] > lower_quantile:
                lower_idx -= 1
            upper_idx = np.searchsorted(cdf_probs, upper_quantile)
        else:
            lower_idx = int(lower_quantile * self.n_samples)
            upper_idx = int(np.ceil(upper_quantile * self.n_samples))
            if upper_idx == self.n_samples:
                upper_idx -= 1

        lower_bound = self.preds[lower_idx, range(self.n_inputs)].T
        upper_bound = self.preds[upper_idx, range(self.n_inputs)].T

        if not calibrated:
            # this won't necessarily hold after calibration because preds are drawn from the
            # uncalibrated distribution
            assert (
                ((lower_bound <= self.preds) & (self.preds <= upper_bound)).mean(axis=0)
                >= conf_level
            ).all()
        return lower_bound, upper_bound

    @staticmethod
    def _get_confidence_interval_cdf(conf_level: float = 0.95) -> Tuple[float, float]:
        """
        Compute the lower and upper bounds on the CDF for `conf_level`.
        While the other confidence interval functions given intervals in the input space,
        this function returns what the CDF probability of the lower/upper bound would be.
        To get to bounds on the input space, you have to use the quantile function to
        find the input that has the appropriate CDF probability.

        The bounds are such that (upper_bound - lower_bound) == conf_level and they're
        centered around 0.5. For example, if `conf_level == 0.9` then `lower_bound == 0.05`
        and `upper_bound == 0.95`.
        :returns: lower_bound, upper_bound
        """
        lower_bound = (1 - conf_level) / 2
        upper_bound = 1 - lower_bound
        assert np.isclose(upper_bound - lower_bound, conf_level)
        lower_bound = round(lower_bound, 10)
        upper_bound = round(upper_bound, 10)
        return lower_bound, upper_bound

    @staticmethod
    def compute_aucc(
        expected_conf: np.ndarray,
        observed_conf: np.ndarray,
        zero_center: bool = True,
        scale_to_one: bool = True,
    ) -> float:
        """
        Compute the area under the calibration curve (AUCC).
        An AUCC of 0.5 corresponds to perfect calibration (or AUCC = 0 if zero-centered).
        A higher AUCC means underconfidence; lower means overconfidence.

        :param zero_center: normally the AUCC is in [0, 1] with 0.5 being optimal. Zero-centering
          subtracts 0.5 so that the AUCC is in [-0.5, 0.5] with 0 being optimal.
        :param scale_to_one: if the AUCC is zero-centered, setting this to True multiples by 2
          so that the AUCC is in [-1, 1]. If `zero_center == False`, this argument is ignored.
        """

        aucc = np.trapz(observed_conf, expected_conf)
        if zero_center:
            aucc -= 0.5
            if scale_to_one:
                aucc *= 2
        return aucc

    def plot_calibration_curve(self, labels, calibrated: bool = False, **plot_kwargs):
        import matplotlib.pyplot as plt

        expected_conf, observed_conf = self.get_calibration_curve(labels, calibrated)
        plt.plot(expected_conf, observed_conf, **plot_kwargs)
        plt.xlabel("Expected Confidence Level")
        plt.ylabel("Observed Confidence Level")
        plt.title("Calibration Curve")
