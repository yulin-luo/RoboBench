"""Path utilities."""

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, creating if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
