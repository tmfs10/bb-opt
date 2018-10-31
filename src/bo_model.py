from abc import ABC, abstractmethod, abstractclassmethod
from typing import Optional, Callable, Union, Tuple, Any, Type, TypeVar
import torch
import numpy as np

_BOM = TypeVar("BOM", bound="BOModel")


class BOModel(ABC):
    """
    A BOModel should support all data being passed in as `np.ndarray`s and should
    return data (e.g. predictions) of this type to increase interoparability.

    BOModel is safe for multiple inheritance; inherit from it first in case the
    other parent class is not.
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.device = "cpu"

    @abstractclassmethod
    def get_model(cls: Type[_BOM], n_inputs: int, device="cpu") -> _BOM:
        pass

    @abstractmethod
    def train_model(
        self,
        inputs: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        n_epochs: int,
        batch_size: int,
        optimizer,
    ) -> None:
        pass

    @abstractmethod
    def predict(self, inputs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        pass

    @abstractmethod
    def save_model(self, fname: str, optimizer: Optional = None) -> None:
        pass

    @abstractclassmethod
    def load_model(
        cls: Type[_BOM],
        fname: str,
        device="cpu",
        optimizer_func: Optional[Callable] = None,
    ) -> Union[_BOM, Tuple[_BOM, Any]]:
        pass

    @abstractmethod
    def reset(self):
        """
        Reset all parameters to random initialization values.

        This allows, e.g., doing Bayesian optimization with full retraining
        without needing a new instance for each iteration.
        """
        pass

    def _to_tensor(self, array: Union[np.ndarray, torch.Tensor]):
        if not isinstance(array, torch.Tensor):
            array = torch.tensor(array, device=self.device)
        return array
