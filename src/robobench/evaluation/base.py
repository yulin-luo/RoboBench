"""Base evaluator class and registry.

All evaluators inherit from BaseEvaluator and are registered
the evaluator_registry for dynamic dispatch.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators.

    Subclasses must implement evaluate() which takes a list of
    result dictionaries and returns a score dictionary.
    """

    name: str = "base"

    @abstractmethod
    def evaluate(
        self, results: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate a list of model responses.

        Args:
            results: List of result dicts with 'response', 'id', etc.
            config: Optional evaluator-specific configuration

        Returns:
            Dictionary with scores and metadata
        """
        raise NotImplementedError

    def evaluate_file(
        self, file_path: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate results from a JSON file."""
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        return self.evaluate(results, config)


# Registry for dynamic evaluator lookup
_evaluator_registry: Dict[str, type] = {}


def register_evaluator(name: str):
    """Decorator to register an evaluator class."""
    def decorator(cls: type):
        _evaluator_registry[name] = cls
        return cls
    return decorator


def get_evaluator(name: str) -> BaseEvaluator:
    """Get an evaluator instance by name.

    Args:
        name: Evaluator type name (e.g., "multi_choice", "planning")

    Returns:
        Instantiated evaluator
    """
    if name not in _evaluator_registry:
        raise ValueError(
            f"Unknown evaluator '{name}'. Available: {list(_evaluator_registry.keys())}"
        )
    return _evaluator_registry[name]()


def list_evaluators() -> List[str]:
    """List all registered evaluator names."""
    return list(_evaluator_registry.keys())
