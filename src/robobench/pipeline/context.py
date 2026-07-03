"""Run context for pipeline execution.

Provides per-run state management including paths, seeds, and temporary directories.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RunContext:
    """Context object passed through pipeline nodes."""

    run_id: str
    config: Any  # BenchmarkConfig
    seed: int = 0
    _metadata: Dict[str, Any] = field(default_factory=dict, repr=False)

    def get_results_dir(self, model_name: Optional[str] = None) -> Path:
        """Get the results directory for this run."""
        base = Path(self.config.paths.results_root) / self.run_id
        if model_name:
            base = base / model_name.replace("/", "_").replace("-", "_")
        base.mkdir(parents=True, exist_ok=True)
        return base

    def get_cache_dir(self) -> Path:
        """Get the cache directory."""
        path = Path(self.config.paths.cache_dir) / self.run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_temp_path(self, filename: str) -> Path:
        """Get a temporary file path."""
        path = self.get_cache_dir() / filename
        return path

    def get_result_path(self, model_name: str, suffix: str = "raw.json") -> Path:
        """Get a result file path for a model."""
        result_dir = self.get_results_dir(model_name)
        path = result_dir / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def set_metadata(self, key: str, value: Any) -> None:
        """Store metadata for this run."""
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve metadata for this run."""
        return self._metadata.get(key, default)
