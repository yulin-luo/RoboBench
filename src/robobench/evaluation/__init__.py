"""Evaluation module for RoboBench.
"""

# Import evaluators to trigger registration
from . import multi_choice  # noqa: F401
from . import planning  # noqa: F401
from . import point  # noqa: F401
from . import iou  # noqa: F401
from . import trajectory  # noqa: F401
from .base import get_evaluator, list_evaluators

__all__ = ["get_evaluator", "list_evaluators"]
