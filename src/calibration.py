import numpy as np
from scipy.special import erfinv
from sklearn.isotonic import IsotonicRegression
from typing import Tuple


class CDF:
    def __init__(self, preds: np.ndarray, gaussian_approx: bool = False):
        """
        :param preds: samples to use in construction the (e)CDFs; there will be one CDF
          per input (shape n_samples x n_inputs)
        :param gaussian_approx: whether to approximate the CDF as a Gaussian. If not,
          the emprical CDF (eCDF) is used instead
        """
        self.gaussian_approx = gaussian_approx
        self.n_samples, self.n_inputs = preds.shape

        if gaussian_approx:
            self.pred_means = preds.mean(0)
            self.pred_stds = preds.std(0)
        else:
            self.preds = np.sort(preds, axis=0)

        self.calibration_regressor = None

    def get_cdf_probs(self, x: np.ndarray, calibrated: bool = False) -> np.ndarray:
        """
        Get the probability of each element of `x` under the corresponding CDF.
        """
        if self.gaussian_approx:
            cdf_probs = self._get_gaussian_cdf_probs(x)
        cdf_probs = self._get_ecdf_probs(x)

        if not calibrated:
            return cdf_probs

        return self.calibrate_cdf_probs(cdf_probs)

    def train_cdf_calibrator(self, labels: np.ndarray) -> np.ndarray:
        """
        Trains a regressor to calibrate the CDF probabilities based on emprical results `x`.
        Follows the method from https://arxiv.org/pdf/1807.00263.pdf.
        """
        cdf_probs = self.get_cdf_probs(labels).squeeze()
        empirical_cdf_probs = []
        for cdf_prob in cdf_probs:
            empirical_cdf_probs.append((cdf_probs <= cdf_prob).mean())
        empirical_cdf_probs = np.array(empirical_cdf_probs)

        calibration_regressor = IsotonicRegression(
            y_min=0, y_max=1, out_of_bounds="clip"
        )
        calibrated_cdf_probs = calibration_regressor.fit_transform(
            cdf_probs, empirical_cdf_probs
        )
        self.calibration_regressor = calibration_regressor
        return calibrated_cdf_probs

    def calibrate_cdf_probs(self, cdf_probs: np.ndarray) -> np.ndarray:
        """
        :param cdf_probs: (shape: any; it will be `ravel`ed before being calibrated)
        :returns: (shape: `cdf_probs.shape`)
        """
        assert (
            self.calibration_regressor is not None
        ), "A calibrator must be trained by `train_cdf_calibrator` first."
        return self.calibration_regressor.transform(cdf_probs.ravel()).reshape(
            cdf_probs.shape
        )

    def get_confidence_intervals(
        self, conf_level: float = 0.95, calibrated: bool = False
    ):
        """
        Compute the lower and upper bounds of a credible region containing `conf_level` probability mass.
        """
        if self.gaussian_approx:
            return self._get_confidence_interval_gaussian(conf_level, calibrated)
        return self._get_confidence_interval_ecdf(conf_level, calibrated)

    def sample(self, n_samples: int = 1, calibrated: bool = False) -> np.ndarray:
        """
        Sample from the CDF using inversion sampling.

        :param n_samples: number of samples to draw from each CDF.
        :returns: array with `n_samples` samples from each CDF (shape n_inputs x n_samples)
        """
        if self.gaussian_approx:
            return self._sample_gaussian(n_samples, calibrated)
        return self._sample_ecdf(n_samples, calibrated)

    def _get_gaussian_cdf_probs(self, x: np.ndarray) -> np.ndarray:
        # TODO - support 2D x like eCDF does
        assert x.ndim == 1, "Gaussian CDF only supports 1D queries right now."
        cdf_probs = 0.5 * (
            1 + np.erf((x - self.pred_means) / (self.pred_stds * np.sqrt(2.0)))
        )
        return cdf_probs

    def _get_ecdf_probs(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the probability of each element of x under the eCDF for the corresponding input.

        :param x: (shape n_inputs or n_inputs x n_queries)
        :returns: (shape n_inputs x n_queries)
        """
        x = x if x.ndim == 2 else x[:, None]  # add n_queries dimension
        ecdf_idx = np.array(
            [np.searchsorted(self.preds[:, i], x[i, :]) for i in range(self.n_inputs)]
        )
        ecdf_probs = ecdf_idx / self.n_samples
        return ecdf_probs

    def _sample_gaussian(
        self, n_samples: int = 1, calibrated: bool = False
    ) -> np.ndarray:
        # TODO: implement this; could try like below with test points or do a vectorized binary search
        # where you halve the bounds on all points at each step
        raise NotImplementedError

        # test_x = np.array([np.linspace(min_i, max_i, n_test_points) for min_i, max_i in zip(self.preds.min(0) - 1, self.preds.max(0) + 1)])
        # cdf_probs = self.get_cdf_probs(test_x)

        # if calibrated:
        #     cdf_probs = self.calibrate_cdf_probs(cdf_probs)

        # uniform_samples = np.random.uniform(size=(self.n_inputs, n_samples))
        # # find the index of the largest x value which has CDF prob <= the uniform sample
        # max_idx_le_uniform_sample = ((cdf_probs[:, :, None] <= uniform_samples[:, None, :]).argmin(1) - 1)
        # return test_x[range(len(test_x)), max_idx_le_uniform_sample.T].T

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
        lower_quantile, upper_quantile = self._get_confidence_interval_cdf(conf_level)

        if calibrated:
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
        return lower_bound, upper_bound
