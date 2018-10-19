from abc import ABC, abstractmethod, abstractclassmethod
from typing import Optional, Callable, Union, Tuple, Any, Type, TypeVar
import torch

_BOM = TypeVar("BOM", bound="BOModel")


class BOModel(ABC):
    @abstractclassmethod
    def get_model(cls: Type[_BOM], n_inputs: int, device="cpu") -> _BOM:
        pass

    @abstractmethod
    def train_model(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        n_epochs: int,
        batch_size: int,
        optimizer,
    ) -> None:
        pass

    @abstractmethod
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
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
