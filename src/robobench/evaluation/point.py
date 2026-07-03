"""Point coordinate evaluator for affordance reasoning tasks.

Evaluates model predictions for single-point tasks by comparing
predicted (x, y) coordinates against ground truth using Euclidean distance.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseEvaluator, register_evaluator


@register_evaluator("point")
class PointEvaluator(BaseEvaluator):
    """Evaluator for single-point coordinate predictions."""

    name = "point"

    def evaluate(
        self, results: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate point predictions.

        Args:
            results: List of result dicts with 'response' and 'gt_answer'
            config: Optional config (not used for point eval)

        Returns:
            Dictionary with distances and statistics
        """
        distances = []
        details = []

        for item in results:
            if not item or not isinstance(item, dict):
                continue

            response = item.get("response", "")
            gt_answer = item.get("gt_answer", "")

            if not response or not gt_answer:
                continue

            pred_point = self._extract_point(response)
            gt_point = self._parse_gt_point(gt_answer)

            if pred_point and gt_point:
                dist = self._euclidean_distance(pred_point, gt_point)
                distances.append(dist)
                details.append({
                    "id": item.get("id", item.get("request_id", "")),
                    "pred": pred_point,
                    "gt": gt_point,
                    "distance": dist,
                })
            else:
                details.append({
                    "id": item.get("id", item.get("request_id", "")),
                    "pred": pred_point,
                    "gt": gt_point,
                    "distance": None,
                    "error": "Failed to extract point",
                })

        avg_distance = sum(distances) / len(distances) if distances else 0

        return {
            "total": len(results),
            "valid": len(distances),
            "average_distance": round(avg_distance, 4),
            "distances": distances,
            "details": details,
        }

    @staticmethod
    def _extract_point(response: str) -> Optional[Tuple[float, float]]:
        """Extract (x, y) coordinates from model response."""
        # Look for patterns like (0.5, 0.3) or [0.5, 0.3]
        patterns = [
            r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)",
            r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]",
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return float(match.group(1)), float(match.group(2))
        return None

    @staticmethod
    def _parse_gt_point(gt_answer: str) -> Optional[Tuple[float, float]]:
        """Parse ground truth answer string into coordinates."""
        patterns = [
            r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)",
            r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]",
        ]
        for pattern in patterns:
            match = re.search(pattern, gt_answer)
            if match:
                return float(match.group(1)), float(match.group(2))
        return None

    @staticmethod
    def _euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Compute Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
