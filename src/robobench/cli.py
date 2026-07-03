"""Command-line interface for RoboBench.

Entry points:
    robobench inference   -- Run model inference
    robobench evaluate    -- Run evaluation on results
    robobench pipeline    -- Run full benchmark pipeline
    robobench report      -- Generate reports from existing results
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from robobench.config import BenchmarkConfig
from robobench.pipeline.context import RunContext
from robobench.pipeline.executor import PipelineExecutor
from robobench.pipeline.node import (
    AggregateScoresNode,
    BuildPromptsNode,
    EvaluateNode,
    LoadDatasetNode,
    RunInferenceNode,
)


def _load_config(args) -> BenchmarkConfig:
    config_path = args.config
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    return BenchmarkConfig.from_yaml(config_path)


def cmd_inference(args):
    """Run inference for specified models and dimensions."""
    config = _load_config(args)
    run_id = args.run_id or "run_0"
    context = RunContext(run_id=run_id, config=config, seed=config.get_seed(0))

    dimensions = config.get_enabled_dimensions()
    models = config.get_model_names()

    if args.model:
        models = [m for m in models if m in args.model]
    if args.dimension:
        dimensions = {k: v for k, v in dimensions.items() if k in args.dimension}

    print(f"Inference: {len(models)} models x {len(dimensions)} dimensions")

    for dim_name, dim_config in dimensions.items():
        for model_name in models:
            print(f"\n{'=' * 60}")
            print(f"Dimension: {dim_name} | Model: {model_name}")
            print(f"{'=' * 60}")

            executor = PipelineExecutor(context)
            executor.add_node(LoadDatasetNode)
            executor.add_node(BuildPromptsNode)
            executor.add_node(RunInferenceNode)

            initial = {
                "dimension": dim_name,
                "subtask": args.subtask or "",
                "dimension_config": dim_config,
                "model_name": model_name,
                "vision": not args.text_only,
                "max_samples": args.max_samples,
            }
            executor.run(initial)


def cmd_evaluate(args):
    """Run evaluation on existing results."""
    config = _load_config(args)
    dimensions = config.get_enabled_dimensions()

    if args.dimension:
        dimensions = {k: v for k, v in dimensions.items() if k in args.dimension}

    for dim_name, dim_config in dimensions.items():
        print(f"\nEvaluating: {dim_name} ({dim_config.eval_type})")

        eval_type = dim_config.eval_type
        eval_config = config.evaluation

        # Determine evaluator-specific config
        if eval_type == "planning":
            eval_cfg = eval_config.planning.model_dump()
        elif eval_type == "multi_choice":
            eval_cfg = eval_config.multi_choice.model_dump()
        else:
            eval_cfg = {}

        # Find result files for this dimension.
        results_dir = Path(config.paths.results_root)
        raw_paths = sorted(results_dir.rglob("raw.json"))
        if not raw_paths:
            print(f"  No raw.json files found under {results_dir}")

        for raw_path in raw_paths:
            print(f"  Evaluating file: {raw_path}")

            context = RunContext(run_id="eval", config=config)
            executor = PipelineExecutor(context)
            executor.add_node(EvaluateNode)

            inputs = {
                "raw_responses": str(raw_path),
                "eval_type": eval_type,
                "eval_config": eval_cfg,
            }
            executor.run(inputs)


def cmd_pipeline(args):
    """Run full benchmark pipeline end-to-end."""
    config = _load_config(args)
    repeats = args.repeats or config.runs.num_repeats

    if args.dry_run:
        executor = PipelineExecutor(RunContext(run_id="dry_run", config=config))
        executor.add_node(LoadDatasetNode)
        executor.add_node(BuildPromptsNode)
        executor.add_node(RunInferenceNode)
        executor.add_node(EvaluateNode)
        if repeats > 1:
            executor.add_node(AggregateScoresNode)
        print("Pipeline graph:")
        for node_name in executor.dry_run():
            print(f"  - {node_name}")
        return

    print(f"Running full pipeline with {repeats} repeats")

    for run_idx in range(repeats):
        run_id = f"run_{run_idx}"
        seed = config.get_seed(run_idx)
        context = RunContext(run_id=run_id, config=config, seed=seed)

        print(f"\n{'=' * 60}")
        print(f"Run {run_idx + 1}/{repeats}: {run_id} (seed={seed})")
        print(f"{'=' * 60}")

        dimensions = config.get_enabled_dimensions()
        models = config.get_model_names()

        for dim_name, dim_config in dimensions.items():
            for model_name in models:
                print(f"\n[{dim_name}] [{model_name}]")

                executor = PipelineExecutor(context)
                executor.add_node(LoadDatasetNode)
                executor.add_node(BuildPromptsNode)
                executor.add_node(RunInferenceNode)
                executor.add_node(EvaluateNode)

                eval_type = dim_config.eval_type
                if eval_type == "planning":
                    eval_cfg = config.evaluation.planning.model_dump()
                elif eval_type == "multi_choice":
                    eval_cfg = config.evaluation.multi_choice.model_dump()
                else:
                    eval_cfg = {}

                initial = {
                    "dimension": dim_name,
                    "subtask": args.subtask or "",
                    "dimension_config": dim_config,
                    "model_name": model_name,
                    "vision": True,
                    "eval_type": eval_type,
                    "eval_config": eval_cfg,
                    "max_samples": args.max_samples,
                }
                executor.run(initial)

    # Aggregate scores across runs
    if repeats > 1:
        print(f"\n{'=' * 60}")
        print("Aggregating scores across runs...")
        print(f"{'=' * 60}")

        context = RunContext(run_id="aggregate", config=config)
        executor = PipelineExecutor(context)
        executor.add_node(AggregateScoresNode)

        # Collect scores from all runs
        run_scores = []
        for run_idx in range(repeats):
            run_dir = Path(config.paths.results_root) / f"run_{run_idx}"
            if run_dir.exists():
                # Collect scores from evaluated results
                scores = {}
                for eval_file in run_dir.rglob("evaluated.json"):
                    with open(eval_file, "r") as f:
                        data = json.load(f)
                        # Extract dimension and score
                        dim_name = eval_file.parent.parent.name  # heuristic
                        scores[dim_name] = data
                run_scores.append(scores)

        executor.run({"run_scores": run_scores})


def cmd_report(args):
    """Generate reports from existing results."""
    config = _load_config(args)
    print("Report generation: TODO")


def main():
    parser = argparse.ArgumentParser(
        prog="robobench",
        description="RoboBench: Benchmark MLLMs on robotic manipulation tasks",
    )
    parser.add_argument(
        "--config",
        default="config/benchmark.yaml",
        help="Path to benchmark configuration YAML file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # inference command
    p_inference = subparsers.add_parser("inference", help="Run model inference")
    p_inference.add_argument("--model", nargs="+", help="Specific models to run")
    p_inference.add_argument(
        "--dimension", nargs="+", help="Specific dimensions to run"
    )
    p_inference.add_argument("--subtask", help="Specific subtask name/file fragment to run")
    p_inference.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of questions to load per selected dimension",
    )
    p_inference.add_argument("--run-id", help="Run identifier")
    p_inference.add_argument(
        "--text-only", action="store_true", help="Run without vision (text-only ablation)"
    )

    # evaluate command
    p_evaluate = subparsers.add_parser("evaluate", help="Evaluate model responses")
    p_evaluate.add_argument(
        "--dimension", nargs="+", help="Specific dimensions to evaluate"
    )

    # pipeline command
    p_pipeline = subparsers.add_parser("pipeline", help="Run full benchmark pipeline")
    p_pipeline.add_argument(
        "--repeats", type=int, help="Number of repeated runs (overrides config)"
    )
    p_pipeline.add_argument("--subtask", help="Specific subtask name/file fragment to run")
    p_pipeline.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of questions to load per selected dimension",
    )
    p_pipeline.add_argument(
        "--dry-run", action="store_true", help="Show pipeline graph without executing"
    )

    # report command
    p_report = subparsers.add_parser("report", help="Generate reports")
    p_report.add_argument(
        "--output", "-o", default=".", help="Output directory for reports"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "inference":
        cmd_inference(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "pipeline":
        cmd_pipeline(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
