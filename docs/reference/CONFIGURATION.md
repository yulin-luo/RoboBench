# Benchmark configuration

Copy [the benchmark configuration template](../../config/benchmark.example.yaml) to `config/benchmark.yaml`. See the [run guide](../guides/QUICKSTART.md) for environment variables and inference commands.

## API Settings

| Field | Description |
| --- | --- |
| `api.base_url` | OpenAI-compatible endpoint for the model being evaluated |
| `api.api_key` | API key for the model endpoint, usually supplied as `${DUBRIFY_API_KEY}` |
| `api.api_max_concurrent` | Request-level API concurrency |
| `api.task_timeout` | Per-request timeout in seconds |
| `api.retry_attempts` | Maximum retry attempts for transient failures |
| `evaluation.planning.eval_model` | Model used to judge Planning Q1/Q2/Q3 responses |
| `evaluation.api` | Independent OpenAI-compatible API configuration for Planning judging and optional MCQ answer normalization |

## Model Selection

```yaml
models:
  - name: "gpt-5.4"
    provider: "openai"
    vision: true

text_only_variants:
  - name: "gpt-5.4"
    suffix: "text_only"
```

## Dimension Selection

```yaml
dimensions:
  perception_reasoning:
    enabled: true
    eval_type: "multi_choice"
    subtasks:
      - static_attribute
      - spatial_relation
```

## Paths

| Field | Description |
| --- | --- |
| `paths.data_root` | Local RoboBench dataset directory containing `system_prompt.json`, released `questions.json` files, and images |
| `paths.results_root` | Model outputs and evaluated scores |
| `paths.cache_dir` | Checkpoints and temporary files |
