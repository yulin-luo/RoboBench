from copy import deepcopy

import pytest

from robobench.config import BenchmarkConfig


@pytest.fixture
def config_data(tmp_path):
    return {
        "api": {
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "test-key",
            "extra_body": {},
            "max_tokens": 128,
            "api_max_concurrent": 2,
            "task_timeout": 30,
            "retry_attempts": 2,
            "retry_backoff": {"multiplier": 1, "min": 1, "max": 2},
        },
        "models": [{"name": "test-model", "provider": "openai", "vision": True}],
        "text_only_variants": [],
        "runs": {"num_repeats": 1, "seed_strategy": "fixed", "skip_existing": True},
        "paths": {
            "data_root": str(tmp_path / "data"),
            "results_root": str(tmp_path / "results"),
            "cache_dir": str(tmp_path / "cache"),
        },
        "dimensions": {
            "perception_reasoning": {
                "enabled": True,
                "eval_type": "multi_choice",
                "subtasks": ["static_attribute", "tool_usage"],
            }
        },
        "evaluation": {
            "api": {
                "base_url": "http://127.0.0.1:8001/v1",
                "api_key": "judge-key",
                "extra_body": {},
                "max_tokens": 128,
                "api_max_concurrent": 2,
                "task_timeout": 30,
                "retry_attempts": 2,
                "retry_backoff": {"multiplier": 1, "min": 1, "max": 2},
            },
            "planning": {
                "eval_model": "judge-model",
            },
            "multi_choice": {"normalize_with_gpt": False, "gpt_model": "judge-model"},
        },
        "output": {
            "save_raw_responses": True,
            "report_formats": ["json"],
            "per_task_breakdown": True,
        },
    }


@pytest.fixture
def benchmark_config(config_data):
    return BenchmarkConfig.model_validate(deepcopy(config_data))
