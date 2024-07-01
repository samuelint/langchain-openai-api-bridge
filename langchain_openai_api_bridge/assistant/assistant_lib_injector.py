from abc import ABC, abstractmethod
from typing import Literal, Optional, Type, TypeVar

T = TypeVar("T")


class BaseAssistantLibInjector(ABC):
    @abstractmethod
    def get(self, cls: Type[T]) -> T:
        pass

    @abstractmethod
    def register(
        self,
        cls: Type[T],
        to: Optional[T] = None,
        scope: Literal["singleton", None] = None,
    ) -> None:
        pass
