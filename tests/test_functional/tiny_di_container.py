from typing import Literal, Optional, Type, TypeVar, Dict, Any, Union
import inspect

from langchain_openai_api_bridge.assistant.assistant_lib_injector import (
    BaseAssistantLibInjector,
)

T = TypeVar("T")


# !! This DI Implementation is an example and should not be used in production !!
# Prefer using a well implemented DI Container, like
# https://github.com/python-injector/injector
# or
# https://python-dependency-injector.ets-labs.org
class TinyDIContainer:
    def __init__(self):
        self.services: Dict[Type[Any], Any] = {}
        self.singletons: Dict[Type[Any], Any] = {}
        self.implementations: Dict[Type[Any], Type[Any]] = {}

    def register(
        self,
        cls: Type[T],
        service: Union[T, None] = None,
        *,
        singleton: bool = False,
        to: Type[T] = None
    ) -> None:
        implementation = to if to else cls
        if singleton:
            self.singletons[cls] = service if service else implementation
        else:
            self.services[cls] = service if service else implementation

    def resolve(self, cls: Type[T]) -> T:
        if cls in self.singletons:
            if isinstance(self.singletons[cls], type):
                # Create and store the singleton instance
                self.singletons[cls] = self._create_instance(self.singletons[cls])
            return self.singletons[cls]

        if cls in self.services:
            if isinstance(self.services[cls], type):
                # Create a new instance every time it's resolved
                return self._create_instance(self.services[cls])
            return self.services[cls]

        if cls in self.implementations:
            return self._create_instance(self.implementations[cls])

        # Automatic resolution of dependencies
        return self._create_instance(cls)

    def _create_instance(self, cls: Type[T]) -> T:
        constructor = inspect.signature(cls.__init__)
        dependencies = {
            name: self.resolve(param.annotation)
            for name, param in constructor.parameters.items()
            if name != "self" and param.annotation != param.empty
        }
        return cls(**dependencies)


class AssistantLibInjector(BaseAssistantLibInjector):
    def __init__(self):
        self.di_container = TinyDIContainer()

    def get(self, cls: Type[T]) -> T:
        return self.di_container.resolve(cls=cls)

    def register(
        self,
        cls: Type[T],
        to: Optional[T] = None,
        scope: Literal["singleton", None] = None,
    ) -> None:
        self.di_container.register(cls=cls, service=to, singleton=scope == "singleton")
