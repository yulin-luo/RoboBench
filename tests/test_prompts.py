import json

import pytest

from robobench.prompts.builder import PromptBuilder


def test_prompt_loader_rejects_malformed_jsonl(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"request_id": "ok"}) + "\nnot-json\n", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        PromptBuilder.load(path)


def test_prompt_builder_rejects_unsupported_image_mode():
    with pytest.raises(ValueError, match="only supports base64"):
        PromptBuilder().build([], mode="url")
