"""Configuration management for RoboBench.

Loads and validates benchmark.yaml using Pydantic models.
Supports environment variable substitution (e.g., ${DUBRIFY_API_KEY}).
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ${VAR} patterns with environment variables."""
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([^}]+)\}")

        def replacer(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return pattern.sub(replacer, value)
    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


class APIConfig(BaseModel):
    """API calling configuration."""

    base_url: str = "https://dubrify.com/v1"
    api_key: str = ""
    max_concurrent: int = 10
    api_max_concurrent: int = 10
    task_timeout: int = 960
    retry_attempts: int = 10
    retry_backoff: Dict[str, float] = Field(default_factory=lambda: {"multiplier": 1, "min": 1, "max": 10})


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str
    provider: str = "openai"
    vision: bool = True


class RunConfig(BaseModel):
    """Run configuration for repeat experiments."""

    num_repeats: int = 3
    seed_strategy: Literal["incremental", "fixed", "random"] = "incremental"
    skip_existing: bool = True


class PathConfig(BaseModel):
    """Path configuration."""

    data_root: str = "data/RoboBench-hf"
    middle_file_dir: str = "data/middle_file"
    results_root: str = "results"
    cache_dir: str = "cache"
    # Prefix embedded in the released dataset's image paths; rewritten to `new_prefix`
    # (falls back to `data_root`) so images resolve against your local dataset copy.
    old_prefix: str = "/share/project/test/robobench/robobench/RoboBench-hf"
    new_prefix: str = ""

    @field_validator("data_root", "middle_file_dir", "results_root", "cache_dir")
    @classmethod
    def _resolve_path(cls, v: str) -> str:
        v = os.path.expanduser(v)
        # If relative, resolve relative to config file directory (set at load time)
        return v


class DimensionConfig(BaseModel):
    """Configuration for a single evaluation dimension."""

    enabled: bool = True
    eval_type: Literal["multi_choice", "planning", "point", "iou", "trajectory"] = "multi_choice"
    system_prompt_key: str = ""
    subtasks: List[str] = Field(default_factory=list)


class PlanningEvalConfig(BaseModel):
    """Planning evaluation-specific configuration."""

    eval_model: str = "gpt-4o"
    use_cache: bool = True


class MultiChoiceEvalConfig(BaseModel):
    """Multi-choice evaluation-specific configuration."""

    normalize_with_gpt: bool = True
    gpt_model: str = "gpt-4o"


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    planning: PlanningEvalConfig = Field(default_factory=PlanningEvalConfig)
    multi_choice: MultiChoiceEvalConfig = Field(default_factory=MultiChoiceEvalConfig)


class OutputConfig(BaseModel):
    """Output configuration."""

    save_raw_responses: bool = True
    report_formats: List[str] = Field(default_factory=lambda: ["json", "csv", "markdown"])
    per_task_breakdown: bool = True


class BenchmarkConfig(BaseModel):
    """Top-level benchmark configuration."""

    api: APIConfig = Field(default_factory=APIConfig)
    models: List[ModelConfig] = Field(default_factory=list)
    text_only_variants: List[Dict[str, str]] = Field(default_factory=list)
    runs: RunConfig = Field(default_factory=RunConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    dimensions: Dict[str, DimensionConfig] = Field(default_factory=dict)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

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
        if isinstance(substituted.get("paths"), dict):
            for key in ["data_root", "middle_file_dir", "results_root", "cache_dir", "new_prefix"]:
                if key in substituted["paths"]:
                    val = substituted["paths"][key]
                    if val and not val.startswith("/"):
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
