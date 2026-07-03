"""Path resolution utilities.

Replaces hardcoded path prefixes with configurable ones.
"""

import os
from pathlib import Path


def resolve_image_path(img_path: str, old_prefix: str = "", new_prefix: str = "") -> str:
    """Resolve image path by replacing old hardcoded prefix with current data directory."""
    if old_prefix and img_path.startswith(old_prefix):
        return img_path.replace(old_prefix, new_prefix, 1)
    return img_path


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, creating if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_middle_file_path(data_root: str, dimension: str, subtask: str) -> Path:
    """Construct path to a middle file given dimension and subtask."""
    # Map dimension/subtask to middle file naming convention
    base = Path(data_root) / "middle_file"
    # Try multiple naming patterns
    patterns = [
        f"{dimension}_{subtask}_questions.jsonl",
        f"processed_question_{subtask}.jsonl",
        f"{subtask}.jsonl",
    ]
    for pattern in patterns:
        candidate = base / pattern
        if candidate.exists():
            return candidate
    # Return first pattern as default (will fail gracefully downstream)
    return base / patterns[0]
