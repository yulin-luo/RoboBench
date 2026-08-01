"""Checkpoint / resume mechanism for long-running inference.

Saves intermediate results so that if the process crashes,
it can resume from where it left off.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


class CheckpointManager:
    """Manages checkpoint files for resumable inference.

    Saves a JSON file containing the current state of processed items.
    On resume, reads the checkpoint and skips already-completed items.
    """

    def __init__(self, checkpoint_path: str | Path):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load existing checkpoint if available."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"results": []}

    def save(self, results: List[Any]) -> None:
        """Save current progress to checkpoint file."""
        self._data["results"] = results

        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get_existing_results(self) -> List[Any]:
        """Return previously saved results (for in-memory resume)."""
        return self._data["results"]

    def clear(self) -> None:
        """Remove checkpoint file."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
