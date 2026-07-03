"""Score aggregation and 3-run averaging for RoboBench.

Aggregates results across multiple runs, computing mean, std,
and per-task statistics for reliable benchmarking.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class RunResult:
    """Result from a single benchmark run."""

    run_id: str
    seed: int
    scores: Dict[str, float]


class RunAggregator:
    """Aggregates scores across multiple benchmark runs.

    Computes per-task statistics (mean, std, min, max) and overall scores.
    """

    def __init__(self):
        self.runs: List[RunResult] = []

    def add_run(self, run_id: str, seed: int, scores: Dict[str, float]) -> None:
        """Add a single run's scores."""
        self.runs.append(RunResult(run_id=run_id, seed=seed, scores=scores))

    def aggregate(self, all_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate scores from multiple runs.

        Args:
            all_scores: List of score dictionaries, one per run

        Returns:
            Dictionary with per-task aggregated statistics
        """
        # Collect all task IDs
        all_task_ids = set()
        for run_scores in all_scores:
            if isinstance(run_scores, dict):
                all_task_ids.update(run_scores.keys())

        aggregated = {}
        for task_id in sorted(all_task_ids):
            scores = []
            for run_scores in all_scores:
                if isinstance(run_scores, dict) and task_id in run_scores:
                    val = run_scores[task_id]
                    if isinstance(val, (int, float)):
                        scores.append(float(val))
                    elif isinstance(val, dict) and "accuracy" in val:
                        scores.append(float(val["accuracy"]))

            if scores:
                aggregated[task_id] = {
                    "scores_per_run": scores,
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "num_runs": len(scores),
                }

        return aggregated

    def compute_statistics(self, aggregated: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall statistics from aggregated scores.

        Args:
            aggregated: Output from aggregate()

        Returns:
            Dictionary with overall mean, std, and per-dimension breakdown
        """
        all_means = [v["mean"] for v in aggregated.values()]

        if not all_means:
            return {"overall_mean": 0.0, "overall_std": 0.0, "num_tasks": 0}

        return {
            "overall_mean": float(np.mean(all_means)),
            "overall_std": float(np.std(all_means)),
            "num_tasks": len(all_means),
            "per_task": {
                k: {"mean": v["mean"], "std": v["std"]}
                for k, v in aggregated.items()
            },
        }

    def compute_variance_breakdown(
        self, aggregated: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute variance breakdown to identify unstable tasks.

        Returns tasks sorted by standard deviation (highest first).
        """
        variances = {
            task_id: stats["std"]
            for task_id, stats in aggregated.items()
            if "std" in stats
        }
        return dict(sorted(variances.items(), key=lambda x: x[1], reverse=True))
