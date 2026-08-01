import json

import pytest

from robobench.inference.checkpoint import CheckpointManager


def test_checkpoint_preserves_results_by_request_id(tmp_path):
    checkpoint = CheckpointManager(tmp_path / "checkpoint.json")
    results = [
        {"id": "completed", "response": "D"},
        {"id": "failed", "response": None, "error": "timeout"},
    ]

    checkpoint.save(results)
    reloaded = CheckpointManager(tmp_path / "checkpoint.json")

    assert reloaded.get_existing_results() == results


def test_malformed_checkpoint_raises(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        CheckpointManager(path)
