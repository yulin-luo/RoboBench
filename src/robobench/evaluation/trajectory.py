"""Trajectory evaluator for multi-point trajectory tasks.

Evaluates trajectory predictions using:
- Discrete Fréchet distance
- Hausdorff distance
- RMSE (Root Mean Square Error)
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseEvaluator, register_evaluator


@register_evaluator("trajectory")
class TrajectoryEvaluator(BaseEvaluator):
    """Evaluator for trajectory predictions."""

    name = "trajectory"

    def evaluate(
        self, results: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate trajectory predictions.

        Args:
            results: List of result dicts with 'response' and 'gt_answer'
            config: Optional config

        Returns:
            Dictionary with trajectory metrics
        """
        frechet_distances = []
        hausdorff_distances = []
        rmses = []
        details = []

        for item in results:
            if not item or not isinstance(item, dict):
                continue

            response = item.get("response", "")
            gt_answer = item.get("gt_answer", "")

            pred_points = self._extract_points(response)
            gt_points = self._extract_points(gt_answer)

            if pred_points and gt_points:
                fd = self._discrete_frechet_distance(pred_points, gt_points)
                hd = self._hausdorff_distance(pred_points, gt_points)
                rmse = self._root_mean_square_error(pred_points, gt_points)

                frechet_distances.append(fd)
                hausdorff_distances.append(hd)
                rmses.append(rmse)

                details.append({
                    "id": item.get("id", item.get("request_id", "")),
                    "pred_points": len(pred_points),
                    "gt_points": len(gt_points),
                    "frechet": round(fd, 4),
                    "hausdorff": round(hd, 4),
                    "rmse": round(rmse, 4),
                })
            else:
                details.append({
                    "id": item.get("id", item.get("request_id", "")),
                    "error": "Failed to extract trajectory points",
                })

        return {
            "total": len(results),
            "valid": len(frechet_distances),
            "average_frechet": round(sum(frechet_distances) / len(frechet_distances), 4) if frechet_distances else 0,
            "average_hausdorff": round(sum(hausdorff_distances) / len(hausdorff_distances), 4) if hausdorff_distances else 0,
            "average_rmse": round(sum(rmses) / len(rmses), 4) if rmses else 0,
            "details": details,
        }

    @staticmethod
    def _extract_points(text: str) -> Optional[List[Tuple[float, float]]]:
        """Extract list of [x, y] points from text."""
        # Look for pattern like [[0.1, 0.2], [0.3, 0.4], ...]
        match = re.search(r"\[\s*(\[\s*[0-9.]+\s*,\s*[0-9.]+\s*\]\s*,?\s*)+\]", text)
        if match:
            try:
                points_str = match.group(0)
                points = eval(points_str)
                return [(float(p[0]), float(p[1])) for p in points]
            except (SyntaxError, ValueError, TypeError):
                pass

        # Fallback: find all individual (x, y) pairs
        pairs = re.findall(r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", text)
        if pairs:
            return [(float(p[0]), float(p[1])) for p in pairs]

        return None

    @staticmethod
    def _discrete_frechet_distance(
        P: List[Tuple[float, float]], Q: List[Tuple[float, float]]
    ) -> float:
        """Compute discrete Fréchet distance between two point sequences."""
        n, m = len(P), len(Q)
        if n == 0 or m == 0:
            return float("inf")

        # Dynamic programming
        ca = [[-1.0] * m for _ in range(n)]

        def _c(i: int, j: int) -> float:
            if ca[i][j] > -1:
                return ca[i][j]
            d = ((P[i][0] - Q[j][0]) ** 2 + (P[i][1] - Q[j][1]) ** 2) ** 0.5
            if i == 0 and j == 0:
                ca[i][j] = d
            elif i > 0 and j == 0:
                ca[i][j] = max(_c(i - 1, 0), d)
            elif i == 0 and j > 0:
                ca[i][j] = max(_c(0, j - 1), d)
            elif i > 0 and j > 0:
                ca[i][j] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), d)
            else:
                ca[i][j] = float("inf")
            return ca[i][j]

        return _c(n - 1, m - 1)

    @staticmethod
    def _hausdorff_distance(
        P: List[Tuple[float, float]], Q: List[Tuple[float, float]]
    ) -> float:
        """Compute symmetric Hausdorff distance."""
        def _directed_hausdorff(A: List[Tuple[float, float]], B: List[Tuple[float, float]]) -> float:
            max_min_dist = 0.0
            for a in A:
                min_dist = min(
                    ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for b in B
                )
                max_min_dist = max(max_min_dist, min_dist)
            return max_min_dist

        return max(_directed_hausdorff(P, Q), _directed_hausdorff(Q, P))

    @staticmethod
    def _root_mean_square_error(
        P: List[Tuple[float, float]], Q: List[Tuple[float, float]]
    ) -> float:
        """Compute RMSE between corresponding points."""
        n = min(len(P), len(Q))
        if n == 0:
            return float("inf")

        squared_errors = [
            (P[i][0] - Q[i][0]) ** 2 + (P[i][1] - Q[i][1]) ** 2
            for i in range(n)
        ]
        return (sum(squared_errors) / n) ** 0.5
