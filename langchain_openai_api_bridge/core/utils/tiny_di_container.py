from typing import Type, TypeVar, Dict, Any, Union, Callable
import inspect

T = TypeVar("T")


class TinyDIContainer:
    def __init__(self):
        self.services: Dict[Type[Any], Any] = {}
        self.singletons: Dict[Type[Any], Any] = {}
        self.implementations: Dict[Type[Any], Type[Any]] = {}

    def register(
        self,
        cls: Type[T],
        service: Union[T, Callable[[], T], None] = None,
        *,
        singleton: bool = False,
        to: Type[T] = None
    ) -> None:
        implementation = to if to else cls
        if singleton:
            self.singletons[cls] = service if service else implementation
        else:
            self.services[cls] = service if service else implementation

        self.implementations[cls] = implementation

    def resolve(self, cls: Type[T]) -> T:
        if cls in self.singletons:
            if isinstance(self.singletons[cls], type):
                # Create and store the singleton instance
                self.singletons[cls] = self._create_instance(self.singletons[cls])
            elif callable(self.singletons[cls]):
                # Call the factory function and store the result
                self.singletons[cls] = self.singletons[cls]()
            return self.singletons[cls]

        if cls in self.services:
            if isinstance(self.services[cls], type):
                # Create a new instance every time it's resolved
                return self._create_instance(self.services[cls])
            elif callable(self.services[cls]):
                # Call the factory function
                return self.services[cls]()
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
