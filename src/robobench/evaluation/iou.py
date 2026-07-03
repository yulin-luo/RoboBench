"""IoU (Intersection over Union) evaluator for bounding box tasks.

Evaluates bounding box predictions using IoU metrics,
including average IoU, pass rates at thresholds, and mAP.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseEvaluator, register_evaluator


@register_evaluator("iou")
class IoUEvaluator(BaseEvaluator):
    """Evaluator for bounding box predictions using IoU."""

    name = "iou"

    def evaluate(
        self, results: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate bounding box predictions.

        Args:
            results: List of result dicts with 'response' and 'gt_answer'
            config: Optional config with 'thresholds' (default [0.5, 0.75, 0.9])

        Returns:
            Dictionary with IoU metrics
        """
        thresholds = [0.5, 0.75, 0.9]
        if config:
            thresholds = config.get("thresholds", thresholds)

        ious = []
        details = []

        for item in results:
            if not item or not isinstance(item, dict):
                continue

            response = item.get("response", "")
            gt_answer = item.get("gt_answer", "")

            pred_box = self._extract_bbox(response)
            gt_box = self._extract_bbox(gt_answer)

            if pred_box and gt_box:
                iou = self._calculate_iou(pred_box, gt_box)
                ious.append(iou)
                details.append({
                    "id": item.get("id", item.get("request_id", "")),
                    "pred": pred_box,
                    "gt": gt_box,
                    "iou": round(iou, 4),
                })
            else:
                details.append({
                    "id": item.get("id", item.get("request_id", "")),
                    "pred": pred_box,
                    "gt": gt_box,
                    "iou": 0.0,
                    "error": "Failed to extract bbox",
                })

        avg_iou = sum(ious) / len(ious) if ious else 0

        # Calculate pass rates at thresholds
        pass_rates = {}
        for t in thresholds:
            passed = sum(1 for iou in ious if iou >= t)
            pass_rates[f"iou@{t}"] = round(passed / len(ious), 4) if ious else 0

        return {
            "total": len(results),
            "valid": len(ious),
            "average_iou": round(avg_iou, 4),
            "pass_rates": pass_rates,
            "ious": ious,
            "details": details,
        }

    @staticmethod
    def _extract_bbox(text: str) -> Optional[Tuple[float, float, float, float]]:
        """Extract [x1, y1, x2, y2] bounding box from text."""
        # Look for patterns like [0.1, 0.2, 0.3, 0.4] or (0.1, 0.2, 0.3, 0.4)
        patterns = [
            r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]",
            r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return tuple(float(match.group(i)) for i in range(1, 5))
        return None

    @staticmethod
    def _calculate_iou(
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float],
    ) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)

        inter_width = max(0, xi_max - xi_min)
        inter_height = max(0, yi_max - yi_min)
        inter_area = inter_width * inter_height

        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0
        return inter_area / union_area
