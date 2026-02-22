"""
Component registry for datasets, models, blockers, calibrators, and clusterers.

Allows plug-and-play registration of new components.
"""

from typing import Dict, Type, Any, Optional, Callable
import importlib


class Registry:
    """
    Generic registry for component classes.
    
    Usage:
        registry = Registry("models")
        registry.register("nars", NarsModel)
        model_cls = registry.get("nars")
        model = model_cls(**params)
    """
    
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}
        self._factories: Dict[str, Callable] = {}
    
    def register(self, name: str, cls: Type = None, factory: Callable = None):
        """
        Register a component class or factory.
        
        Can be used as a decorator:
            @registry.register("my_model")
            class MyModel(BaseModel):
                pass
        """
        def decorator(cls_or_fn):
            if callable(cls_or_fn) and not isinstance(cls_or_fn, type):
                self._factories[name] = cls_or_fn
            else:
                self._registry[name] = cls_or_fn
            return cls_or_fn
        
        if cls is not None:
            self._registry[name] = cls
            return cls
        if factory is not None:
            self._factories[name] = factory
            return factory
        return decorator
    
    def get(self, name: str) -> Type:
        """Get a registered class by name."""
        if name in self._registry:
            return self._registry[name]
        if name in self._factories:
            return self._factories[name]
        raise KeyError(f"'{name}' not found in {self.name} registry. "
                      f"Available: {list(self._registry.keys())}")
    
    def create(self, name: str, **kwargs) -> Any:
        """Create an instance of a registered component."""
        cls_or_factory = self.get(name)
        return cls_or_factory(**kwargs)
    
    def list(self) -> list:
        """List all registered component names."""
        return list(self._registry.keys()) + list(self._factories.keys())
    
    def __contains__(self, name: str) -> bool:
        return name in self._registry or name in self._factories


# Global registries
_registries: Dict[str, Registry] = {}


def get_registry(name: str) -> Registry:
    """Get or create a named registry."""
    if name not in _registries:
        _registries[name] = Registry(name)
    return _registries[name]


# Convenience accessors
datasets = get_registry("datasets")
models = get_registry("models")
blockers = get_registry("blockers")
calibrators = get_registry("calibrators")
clusterers = get_registry("clusterers")

