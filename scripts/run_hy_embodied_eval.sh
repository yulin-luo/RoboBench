#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run RoboBench-MCQ and RoboBench-Planning style subsets for an OpenAI-compatible
HY-Embodied endpoint/model.

This script does not load HY-Embodied weights directly. Start an
OpenAI-compatible server first, for example the official Hy-Embodied-VLM-1.0
vLLM server, and set ROBOBENCH_MODEL to the served model name.

Usage:
  ROBOBENCH_MODEL=<served-model-name> bash scripts/run_hy_embodied_eval.sh [all|mcq|planning|smoke|dry-run]

Common environment variables:
  ROBOBENCH_MODEL                 Required served model name in config/benchmark.yaml
  ROBOBENCH_CONFIG                Config path (default: config/benchmark.yaml)
  ROBOBENCH_RUN_ID                Run id (default: hy_embodied_run0)
  ROBOBENCH_MAX_SAMPLES           Optional sample cap for debugging

Before running:
  1. pip install -e .
  2. Download the RoboBench dataset.
  3. Copy config/benchmark.example.yaml to config/benchmark.yaml.
  4. Add your served HY-Embodied model name to config.models.
  5. Start the HY model as an OpenAI-compatible endpoint.
  6. Export ROBOBENCH_API_BASE_URL, DUBRIFY_API_KEY, ROBOBENCH_DATA_ROOT,
     ROBOBENCH_RESULTS_ROOT, ROBOBENCH_CACHE_DIR, ROBOBENCH_JUDGE_API_BASE_URL,
     and ROBOBENCH_JUDGE_API_KEY.

Examples:
  ROBOBENCH_MODEL=hy_a3b bash scripts/run_hy_embodied_eval.sh dry-run
  ROBOBENCH_MODEL=hy_a3b bash scripts/run_hy_embodied_eval.sh smoke
  ROBOBENCH_MODEL=hy_a3b bash scripts/run_hy_embodied_eval.sh mcq
  ROBOBENCH_MODEL=hy_a3b bash scripts/run_hy_embodied_eval.sh planning
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODE="${1:-all}"
CONFIG="${ROBOBENCH_CONFIG:-config/benchmark.yaml}"
MODEL="${ROBOBENCH_MODEL:-}"
RUN_ID="${ROBOBENCH_RUN_ID:-hy_embodied_run0}"
MAX_SAMPLES="${ROBOBENCH_MAX_SAMPLES:-}"
MCQ_DIMENSIONS=(perception_reasoning affordance_reasoning error_analysis)
PLANNING_DIMENSIONS=(instruction_comprehension generalized_planning)

case "$MODE" in
  all|mcq|planning|smoke|dry-run) ;;
  *)
    echo "ERROR: unknown mode '$MODE'. Expected all, mcq, planning, smoke, or dry-run." >&2
    usage >&2
    exit 2
    ;;
esac

if [[ "$MODE" != "dry-run" && -z "$MODEL" ]]; then
  echo "ERROR: ROBOBENCH_MODEL is required." >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config file not found: $CONFIG" >&2
  echo "Copy config/benchmark.example.yaml to config/benchmark.yaml first." >&2
  exit 2
fi

if [[ "$MODE" == "smoke" && -z "$MAX_SAMPLES" ]]; then
  MAX_SAMPLES=1
fi

if [[ "$MODE" != "dry-run" ]]; then
python - "$CONFIG" "$MODEL" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
model = sys.argv[2]
data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
names = [item["name"] for item in data["models"]]
if model not in names:
    print(
        f"ERROR: model '{model}' is not listed under models in {config_path}.",
        file=sys.stderr,
    )
    print("Add an entry such as:", file=sys.stderr)
    print("  - name: \"{}\"".format(model), file=sys.stderr)
    print("    provider: \"openai\"", file=sys.stderr)
    print("    vision: true", file=sys.stderr)
    sys.exit(2)
PY
fi

run_robobench() {
  if command -v robobench >/dev/null 2>&1; then
    robobench "$@"
  else
    PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src" python -m robobench.cli "$@"
  fi
}

run_dimension() {
  local dimension="$1"
  local subtask="${2:-}"
  local extra_args=()
  if [[ -n "$MAX_SAMPLES" ]]; then
    extra_args+=(--max-samples "$MAX_SAMPLES")
  fi
  if [[ -n "$subtask" ]]; then
    extra_args+=(--subtask "$subtask")
  fi

  echo
  echo "================================================================"
  echo "RoboBench dimension: $dimension | model: ${MODEL:-<set ROBOBENCH_MODEL>} | run: $RUN_ID"
  echo "================================================================"
  run_robobench --config "$CONFIG" inference \
    --model "$MODEL" \
    --dimension "$dimension" \
    --run-id "$RUN_ID" \
    "${extra_args[@]}"

  local evaluate_args=(
    --dimension "$dimension"
    --model "$MODEL"
    --run-id "$RUN_ID"
  )
  if [[ -n "$subtask" ]]; then
    evaluate_args+=(--subtask "$subtask")
  fi
  run_robobench --config "$CONFIG" evaluate "${evaluate_args[@]}"
}

run_list() {
  local -a dims=("$@")
  for dimension in "${dims[@]}"; do
    run_dimension "$dimension"
  done
}

inspect_list() {
  local -a dims=("$@")
  run_robobench --config "$CONFIG" inspect-data \
    --metadata-only \
    --dimension "${dims[@]}"
}

case "$MODE" in
  mcq)
    run_list "${MCQ_DIMENSIONS[@]}"
    ;;
  planning)
    run_list "${PLANNING_DIMENSIONS[@]}"
    ;;
  smoke)
    run_dimension perception_reasoning static_attribute
    run_dimension instruction_comprehension explicit_object_goal
    ;;
  dry-run)
    inspect_list "${MCQ_DIMENSIONS[@]}" "${PLANNING_DIMENSIONS[@]}"
    ;;
  all)
    run_list "${MCQ_DIMENSIONS[@]}"
    run_list "${PLANNING_DIMENSIONS[@]}"
    ;;
esac
