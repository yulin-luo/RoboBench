# Evaluating HY-Embodied on RoboBench

This page provides a RoboBench-side helper for users who want to evaluate an
OpenAI-compatible HY-Embodied endpoint on the two RoboBench settings mentioned
by the HY-Embodied release:

- `RoboBench-MCQ`
- `RoboBench-Planning`

We have asked the HY-Embodied maintainers to confirm the exact protocol they
used for their reported numbers. Until that is clarified, the helper defaults to
the public RoboBench settings below and keeps the dimensions configurable.
The protocol clarification is tracked at:
https://github.com/Tencent-Hunyuan/HY-Embodied/issues/14#issuecomment-5070118229

## Default Mapping

| Setting | Default RoboBench dimensions | Notes |
| --- | --- | --- |
| `RoboBench-MCQ` | `perception_reasoning affordance_reasoning error_analysis` | Dimensions 2, 4, and 5 use the multiple-choice evaluator. Override with `ROBOBENCH_HY_MCQ_DIMENSIONS` if you need a different protocol. |
| `RoboBench-Planning` | `instruction_comprehension generalized_planning` | Dimensions 1 and 3 use the planning evaluator. Override with `ROBOBENCH_HY_PLANNING_DIMENSIONS` if the target protocol includes additional planning dimensions. |

Planning evaluation can call an evaluator model configured by
`evaluation.planning.eval_model` in `config/benchmark.yaml`.

## Setup

Install RoboBench and download the dataset:

```bash
git clone https://github.com/yulin-luo/RoboBench.git
cd RoboBench
pip install -e .

huggingface-cli download \
  --repo-type dataset LeoFan01/RoboBench \
  --local-dir data/RoboBench-hf
```

Create a local config:

```bash
cp config/benchmark.example.yaml config/benchmark.yaml
```

Edit `config/benchmark.yaml` and add the served HY-Embodied model name under
`models`, for example:

```yaml
models:
  - name: "HY-Embodied-0.5-MoT-2B"
    provider: "openai"
    vision: true
```

Point RoboBench to your OpenAI-compatible endpoint and local paths:

```bash
export ROBOBENCH_API_BASE_URL="http://your-server:port/v1"
export DUBRIFY_API_KEY="your-api-key-or-placeholder"
export ROBOBENCH_DATA_ROOT="$PWD/data/RoboBench-hf"
export ROBOBENCH_MIDDLE_FILE_DIR="$PWD/data/middle_file"
export ROBOBENCH_RESULTS_ROOT="$PWD/results"
export ROBOBENCH_CACHE_DIR="$PWD/cache"
export ROBOBENCH_OLD_IMAGE_PREFIX=""
```

If your endpoint does not require authentication, use any non-empty placeholder
for `DUBRIFY_API_KEY`.

## One-Command Runs

Preview the selected dimensions first:

```bash
ROBOBENCH_MODEL=HY-Embodied-0.5-MoT-2B \
  bash scripts/run_hy_embodied_eval.sh dry-run
```

Then run a two-dimension smoke test with one sample per dimension:

```bash
ROBOBENCH_MODEL=HY-Embodied-0.5-MoT-2B \
  bash scripts/run_hy_embodied_eval.sh smoke
```

Run the MCQ setting:

```bash
ROBOBENCH_MODEL=HY-Embodied-0.5-MoT-2B \
  bash scripts/run_hy_embodied_eval.sh mcq
```

Run the Planning setting:

```bash
ROBOBENCH_MODEL=HY-Embodied-0.5-MoT-2B \
  bash scripts/run_hy_embodied_eval.sh planning
```

Run both settings:

```bash
ROBOBENCH_MODEL=HY-Embodied-0.5-MoT-2B \
  bash scripts/run_hy_embodied_eval.sh all
```

## Protocol Overrides

Use environment variables to match a different reported protocol:

```bash
ROBOBENCH_MODEL=HY-Embodied-0.5-MoT-2B \
ROBOBENCH_HY_MCQ_DIMENSIONS="perception_reasoning affordance_reasoning error_analysis" \
ROBOBENCH_HY_PLANNING_DIMENSIONS="instruction_comprehension generalized_planning" \
ROBOBENCH_RUN_ID=hy_embodied_run0 \
  bash scripts/run_hy_embodied_eval.sh all
```

For quick debugging:

```bash
ROBOBENCH_MODEL=HY-Embodied-0.5-MoT-2B \
ROBOBENCH_MAX_SAMPLES=5 \
  bash scripts/run_hy_embodied_eval.sh all
```

## Outputs

Raw responses and evaluated outputs are written under `paths.results_root` from
`config/benchmark.yaml`, typically `results/`.

Official RoboBench reference score tables and model-output JSON files are
available at:

```text
https://huggingface.co/datasets/lyl010221-pku/RoboBench-Results
```
