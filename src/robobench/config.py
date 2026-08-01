"""Configuration management for RoboBench.

Loads and validates benchmark.yaml using Pydantic models.
Supports environment variable substitution (e.g., ${DUBRIFY_API_KEY}).
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, field_validator


class StrictConfigModel(BaseModel):
    """Reject unknown configuration fields."""

    model_config = ConfigDict(extra="forbid")


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ${VAR} patterns with environment variables."""
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([^}]+)\}")

        def replacer(match):
            var_name = match.group(1)
            return os.environ[var_name]

        return pattern.sub(replacer, value)
    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


class RetryBackoffConfig(StrictConfigModel):
    """API retry backoff configuration."""

    multiplier: float
    min: float
    max: float


class APIConfig(StrictConfigModel):
    """API calling configuration."""

    base_url: str
    api_key: str
    extra_body: Dict[str, Any]
    max_tokens: Optional[int]
    api_max_concurrent: int
    task_timeout: int
    retry_attempts: int
    retry_backoff: RetryBackoffConfig


class ModelConfig(StrictConfigModel):
    """Model configuration."""

    name: str
    provider: str
    vision: bool


class RunConfig(StrictConfigModel):
    """Run configuration for repeat experiments."""

    num_repeats: int
    seed_strategy: Literal["incremental", "fixed", "random"]
    skip_existing: bool


class PathConfig(StrictConfigModel):
    """Path configuration."""

    data_root: str
    results_root: str
    cache_dir: str

    @field_validator("data_root", "results_root", "cache_dir")
    @classmethod
    def _resolve_path(cls, v: str) -> str:
        v = os.path.expanduser(v)
        # If relative, resolve relative to config file directory (set at load time)
        return v


class DimensionConfig(StrictConfigModel):
    """Configuration for a single evaluation dimension."""

    enabled: bool
    eval_type: Literal["multi_choice", "planning", "point", "iou", "trajectory"]
    subtasks: List[str]


class PlanningEvalConfig(StrictConfigModel):
    """Planning evaluation-specific configuration."""

    eval_model: str


class MultiChoiceEvalConfig(StrictConfigModel):
    """Multi-choice evaluation-specific configuration."""

    normalize_with_gpt: bool
    gpt_model: str


class EvaluationConfig(StrictConfigModel):
    """Evaluation configuration."""

    api: APIConfig
    planning: PlanningEvalConfig
    multi_choice: MultiChoiceEvalConfig


class OutputConfig(StrictConfigModel):
    """Output configuration."""

    save_raw_responses: bool
    report_formats: List[str]
    per_task_breakdown: bool


class BenchmarkConfig(StrictConfigModel):
    """Top-level benchmark configuration."""

    api: APIConfig
    models: List[ModelConfig]
    text_only_variants: List[Dict[str, str]]
    runs: RunConfig
    paths: PathConfig
    dimensions: Dict[str, DimensionConfig]
    evaluation: EvaluationConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkConfig":
        """Load configuration from a YAML file with env var substitution.

        Relative paths in 'paths' are resolved relative to the config file directory.
        """
        config_path = Path(path).resolve()
        config_dir = config_path.parent

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        substituted = _substitute_env_vars(raw)

        # Resolve relative paths in 'paths' section
        for key in ["data_root", "results_root", "cache_dir"]:
            val = substituted["paths"][key]
            if not Path(val).is_absolute():
                substituted["paths"][key] = str((config_dir / val).resolve())

        return cls.model_validate(substituted)

    def get_enabled_dimensions(self) -> Dict[str, DimensionConfig]:
        """Return only enabled dimensions."""
        return {k: v for k, v in self.dimensions.items() if v.enabled}

    def get_model_names(self) -> List[str]:
        """Return list of model names to evaluate."""
        return [m.name for m in self.models]

    def get_seed(self, run_id: int) -> int:
        """Compute seed for a given run."""
        if self.runs.seed_strategy == "incremental":
            return run_id
        if self.runs.seed_strategy == "fixed":
            return 42
        # random: will be set at runtime
        return run_id
