"""Checkpoint / resume mechanism for long-running inference.

Saves intermediate results so that if the process crashes,
it can resume from where it left off.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


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
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"completed_ids": [], "results": [], "last_index": -1}

    def save(self, results: List[Any], last_index: int) -> None:
        """Save current progress to checkpoint file."""
        self._data["results"] = results
        self._data["last_index"] = last_index
        # Extract IDs from results that have responses
        completed = []
        for r in results:
            if r and r.get("response") is not None:
                completed.append(r.get("id", ""))
        self._data["completed_ids"] = completed

        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get_resume_index(self) -> int:
        """Return the index to resume from (0 if no checkpoint)."""
        return self._data.get("last_index", -1) + 1

    def get_existing_results(self) -> List[Any]:
        """Return previously saved results (for in-memory resume)."""
        return self._data.get("results", [])

    def is_completed(self, request_id: str) -> bool:
        """Check if a specific request ID has already been processed."""
        return request_id in self._data.get("completed_ids", [])

    def clear(self) -> None:
        """Remove checkpoint file."""
        if self.checkpoint_path.exists():
            os.remove(self.checkpoint_path)
