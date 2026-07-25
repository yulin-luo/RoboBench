#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run RoboBench-MCQ and RoboBench-Planning style subsets for an OpenAI-compatible
HY-Embodied endpoint/model.

Usage:
  ROBOBENCH_MODEL=<served-model-name> bash scripts/run_hy_embodied_eval.sh [all|mcq|planning|smoke|dry-run]

Common environment variables:
  ROBOBENCH_MODEL                 Required served model name in config/benchmark.yaml
  ROBOBENCH_CONFIG                Config path (default: config/benchmark.yaml)
  ROBOBENCH_RUN_ID                Run id (default: hy_embodied_run0)
  ROBOBENCH_MAX_SAMPLES           Optional sample cap for debugging
  ROBOBENCH_HY_MCQ_DIMENSIONS     Default: "perception_reasoning error_analysis"
  ROBOBENCH_HY_PLANNING_DIMENSIONS Default: "generalized_planning"

Before running:
  1. pip install -e .
  2. Download the RoboBench dataset.
  3. Copy config/benchmark.example.yaml to config/benchmark.yaml.
  4. Add your served HY-Embodied model name to config.models.
  5. Export ROBOBENCH_API_BASE_URL, DUBRIFY_API_KEY, ROBOBENCH_DATA_ROOT,
     ROBOBENCH_MIDDLE_FILE_DIR, ROBOBENCH_RESULTS_ROOT, and ROBOBENCH_CACHE_DIR.

Examples:
  ROBOBENCH_MODEL=HY-Embodied-0.5-MoT-2B bash scripts/run_hy_embodied_eval.sh dry-run
  ROBOBENCH_MODEL=HY-Embodied-0.5-MoT-2B bash scripts/run_hy_embodied_eval.sh smoke
  ROBOBENCH_MODEL=HY-Embodied-0.5-MoT-2B bash scripts/run_hy_embodied_eval.sh mcq
  ROBOBENCH_MODEL=HY-Embodied-0.5-MoT-2B bash scripts/run_hy_embodied_eval.sh planning
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
MCQ_DIMENSIONS="${ROBOBENCH_HY_MCQ_DIMENSIONS:-perception_reasoning error_analysis}"
PLANNING_DIMENSIONS="${ROBOBENCH_HY_PLANNING_DIMENSIONS:-generalized_planning}"

case "$MODE" in
  all|mcq|planning|smoke|dry-run) ;;
  *)
    echo "ERROR: unknown mode '$MODE'. Expected all, mcq, planning, smoke, or dry-run." >&2
    usage >&2
    exit 2
    ;;
esac

if [[ -z "$MODEL" ]]; then
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

python - "$CONFIG" "$MODEL" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
model = sys.argv[2]
data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
names = [item.get("name") for item in data.get("models", []) if isinstance(item, dict)]
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

run_robobench() {
  if command -v robobench >/dev/null 2>&1; then
    robobench "$@"
  else
    PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src" python -m robobench.cli "$@"
  fi
}

run_dimension() {
  local dimension="$1"
  local extra_args=()
  if [[ -n "$MAX_SAMPLES" ]]; then
    extra_args+=(--max-samples "$MAX_SAMPLES")
  fi

  echo
  echo "================================================================"
  echo "RoboBench dimension: $dimension | model: $MODEL | run: $RUN_ID"
  echo "================================================================"
  if [[ "$MODE" == "dry-run" ]]; then
    echo "Would run inference and evaluation for this dimension."
    return
  fi

  run_robobench --config "$CONFIG" inference \
    --model "$MODEL" \
    --dimension "$dimension" \
    --run-id "$RUN_ID" \
    "${extra_args[@]}"

  run_robobench --config "$CONFIG" evaluate \
    --dimension "$dimension" \
    --model "$MODEL" \
    --run-id "$RUN_ID"
}

run_list() {
  local dims="$1"
  for dimension in $dims; do
    run_dimension "$dimension"
  done
}

case "$MODE" in
  mcq)
    run_list "$MCQ_DIMENSIONS"
    ;;
  planning)
    run_list "$PLANNING_DIMENSIONS"
    ;;
  smoke)
    run_dimension "$(printf '%s\n' $MCQ_DIMENSIONS | head -n 1)"
    run_dimension "$(printf '%s\n' $PLANNING_DIMENSIONS | head -n 1)"
    ;;
  dry-run)
    run_list "$MCQ_DIMENSIONS"
    run_list "$PLANNING_DIMENSIONS"
    ;;
  all)
    run_list "$MCQ_DIMENSIONS"
    run_list "$PLANNING_DIMENSIONS"
    ;;
esac
