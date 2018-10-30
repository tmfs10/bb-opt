import numpy as np
from abc import ABC, abstractmethod, abstractclassmethod
from typing import Union, Set, List, Optional
from bb_opt.src.bo_model import BOModel
from bb_opt.src.bnn import BNN
from bb_opt.src.deep_ensemble import NNEnsemble
from bb_opt.src.calibration import CDF


class AcquisitionFunction(ABC):
    @abstractmethod
    def acquire(
        self,
        model: BOModel,
        inputs,
        sampled_labels,
        sampled_idx: Set[int],
        batch_size: int,
    ) -> List[int]:
        """
        :param model: (trained) model to use in evaluating points for acquisition
        """
        pass

    def __call__(self, *args, **kwargs):
        return self.acquire(*args, **kwargs)


class GradientAcquisitionFunction(AcquisitionFunction):
    ...


class ExhaustiveAcquisitionFunction(AcquisitionFunction):
    ...


class Uniform(ExhaustiveAcquisitionFunction):
    """
    Acquires (unsampled) inputs uniformly at random.
    """

    def __init__(self, exp: Optional = None):
        """
        :param exp: a CometML experiment to use in logging metrics
        """
        super().__init__()
        self.exp = exp

    @staticmethod
    def acquire(
        model: BOModel, inputs, sampled_labels, sampled_idx: Set[int], batch_size: int
    ) -> List[int]:
        possible_idx = list(range(len(inputs)))
        np.random.shuffle(possible_idx)
        possible_idx = [i for i in possible_idx if i not in sampled_idx]
        acquire_samples = possible_idx[:batch_size]
        return acquire_samples


class ExpectedReward(ExhaustiveAcquisitionFunction):
    """
    The expected reward for a point is the posterior mean for that point.
    The points acquired by this AF are those with highest expected reward.
    """

    def __init__(
        self, exp: Optional = None, n_samples: int = 100, calibrated: bool = False
    ):
        """
        :param exp: a CometML experiment to use in logging metrics
        :param n_bnn_samples: how many samples to draw from the posterior to compute its mean
        :param calibrated: whether to compute the posterior mean from a calibrated distribution or not
        """
        self.exp = exp
        self.n_samples = n_samples
        self.calibrated = calibrated

    def acquire(
        self,
        model: Union[BNN, NNEnsemble],
        inputs,
        sampled_labels,
        sampled_idx: Set[int],
        batch_size: int,
    ) -> List[int]:
        assert isinstance(model, (NNEnsemble, BNN))

        if self.calibrated:
            train_inputs = inputs[list(sampled_idx)]
            non_sampled_idx = set(range(len(inputs))).difference(sampled_idx)
            test_inputs = inputs[list(non_sampled_idx)]

        if isinstance(model, BNN):
            if self.calibrated:
                preds = model.predict(train_inputs)
                cdf_train = CDF(preds)
                cdf_train.train_cdf_calibrator(sampled_labels)

                preds = model.predict(test_inputs)
                cdf_test = CDF(
                    preds, calibration_regressor=cdf_train.calibration_regressor
                )

                means = cdf_test.sample(self.n_samples, self.calibrated).mean(1)
            else:
                means = model.predict(inputs, n_samples=self.n_samples).mean(0)
        elif isinstance(model, NNEnsemble):
            if self.calibrated:
                # TODO: use a Gaussian for NNEnsemble
                preds = model.predict(train_inputs)
                cdf_train = CDF(preds)
                cdf_train.train_cdf_calibrator(sampled_labels)

                preds = model.predict(test_inputs)
                cdf_test = CDF(
                    preds, calibration_regressor=cdf_train.calibration_regressor
                )

                means = cdf_test.sample(self.n_samples, self.calibrated).mean(1)
            else:
                means = model.predict(inputs).mean(0)
        else:
            assert (
                False
            ), "Should never reach this point; update the types allowed in this function."

        sorted_idx = np.argsort(means)
        sorted_idx = [i for i in sorted_idx if i not in sampled_idx]
        acquire_samples = sorted_idx[-batch_size:]  # sorted in descending order

        return acquire_samples
