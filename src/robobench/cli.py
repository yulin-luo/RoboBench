"""Command-line interface for RoboBench.

Entry points:
    robobench inference   -- Run model inference
    robobench evaluate    -- Run evaluation on results
    robobench pipeline    -- Run full benchmark pipeline
    robobench report      -- Generate reports from existing results
"""

import argparse
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


def _raw_path_matches_dimension(raw_path: Path, dimension: str) -> bool:
    task_label = raw_path.parent.name
    return task_label == dimension or task_label.startswith(f"{dimension}_")


def _raw_path_matches_subtask(raw_path: Path, dimension: str, subtask: str | None) -> bool:
    if subtask is None:
        return True
    return raw_path.parent.name == f"{dimension}_{subtask}"


def _safe_label(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def _raw_path_matches_model(raw_path: Path, models: list[str]) -> bool:
    if not models:
        return True
    if len(raw_path.parents) < 2:
        return False
    model_label = raw_path.parent.parent.name
    return model_label in {_safe_label(model) for model in models}


def _raw_path_matches_run(raw_path: Path, results_dir: Path, run_id: str | None) -> bool:
    if run_id is None:
        return True
    if not raw_path.is_relative_to(results_dir):
        return False
    relative = raw_path.relative_to(results_dir)
    return bool(relative.parts) and relative.parts[0] == run_id


def _select_dimensions(config: BenchmarkConfig, requested: list[str] | None):
    dimensions = config.get_enabled_dimensions()
    if requested is None:
        return dimensions
    unknown = sorted(set(requested) - set(dimensions))
    if unknown:
        raise ValueError(f"Unknown or disabled dimensions: {', '.join(unknown)}")
    return {name: dimensions[name] for name in requested}


def _select_models(config: BenchmarkConfig, requested: list[str] | None) -> list[str]:
    models = config.get_model_names()
    if requested is None:
        return models
    unknown = sorted(set(requested) - set(models))
    if unknown:
        raise ValueError(f"Models are not configured: {', '.join(unknown)}")
    return requested


def _select_subtasks(dimension: str, configured: list[str], requested: str | None) -> list[str]:
    if requested is None:
        return configured
    if requested not in configured:
        raise ValueError(f"Unknown subtask '{requested}' for dimension '{dimension}'")
    return [requested]


def cmd_inspect_data(args):
    """Validate released questions, prompts, and image paths."""
    from robobench.data.dataset import RoboBenchDataset

    config = _load_config(args)
    dimensions = _select_dimensions(config, args.dimension)
    if args.subtask is not None and len(dimensions) != 1:
        raise ValueError("--subtask requires exactly one --dimension")

    dataset = RoboBenchDataset(config.paths.data_root)
    total_questions = 0
    total_images = 0
    print(f"Dataset root: {config.paths.data_root}")

    for dimension, dimension_config in dimensions.items():
        subtasks = _select_subtasks(dimension, dimension_config.subtasks, args.subtask)
        for subtask in subtasks:
            questions = dataset.load_questions(dimension, subtask)
            image_paths = [Path(path) for question in questions for path in question["image_urls"]]
            if not args.metadata_only:
                missing_images = [path for path in image_paths if not path.is_file()]
                if missing_images:
                    raise FileNotFoundError(
                        f"Missing {len(missing_images)} images for {dimension}/{subtask}; "
                        f"first missing path: {missing_images[0]}"
                    )
            print(f"  {dimension}/{subtask}: questions={len(questions)} images={len(image_paths)}")
            total_questions += len(questions)
            total_images += len(image_paths)

    print(f"Validated: questions={total_questions} images={total_images}")


def cmd_inference(args):
    """Run inference for specified models and dimensions."""
    config = _load_config(args)
    run_id = args.run_id
    context = RunContext(run_id=run_id, config=config, seed=config.get_seed(0))

    dimensions = _select_dimensions(config, args.dimension)
    models = _select_models(config, args.model)
    if args.subtask is not None and len(dimensions) != 1:
        raise ValueError("--subtask requires exactly one --dimension")

    task_count = sum(
        len(_select_subtasks(dim_name, dim_config.subtasks, args.subtask))
        for dim_name, dim_config in dimensions.items()
    )
    print(f"Inference: {len(models)} models x {task_count} subtasks")

    for dim_name, dim_config in dimensions.items():
        subtasks = _select_subtasks(dim_name, dim_config.subtasks, args.subtask)
        for subtask in subtasks:
            for model_name in models:
                print(f"\n{'=' * 60}")
                print(f"Dimension: {dim_name} | Subtask: {subtask} | Model: {model_name}")
                print(f"{'=' * 60}")

                executor = PipelineExecutor(context)
                executor.add_node(LoadDatasetNode)
                executor.add_node(BuildPromptsNode)
                executor.add_node(RunInferenceNode)

                initial = {
                    "dimension": dim_name,
                    "subtask": subtask,
                    "model_name": model_name,
                    "vision": not args.text_only,
                    "max_samples": args.max_samples,
                }
                executor.run(initial)


def cmd_evaluate(args):
    """Run evaluation on existing results."""
    config = _load_config(args)
    dimensions = _select_dimensions(config, args.dimension)
    if args.subtask is not None and len(dimensions) != 1:
        raise ValueError("--subtask requires exactly one --dimension")

    for dim_name, dim_config in dimensions.items():
        _select_subtasks(dim_name, dim_config.subtasks, args.subtask)
        print(f"\nEvaluating: {dim_name} ({dim_config.eval_type})")

        eval_type = dim_config.eval_type
        # Find result files for this dimension.
        results_dir = Path(config.paths.results_root)
        raw_paths = [
            path
            for path in sorted(results_dir.rglob("raw.json"))
            if _raw_path_matches_dimension(path, dim_name)
            and _raw_path_matches_subtask(path, dim_name, args.subtask)
            and _raw_path_matches_model(path, args.model or [])
            and _raw_path_matches_run(path, results_dir, args.run_id)
        ]
        if not raw_paths:
            print(f"  No raw.json files found for {dim_name} under {results_dir}")

        for raw_path in raw_paths:
            print(f"  Evaluating file: {raw_path}")

            context = RunContext(run_id="eval", config=config)
            executor = PipelineExecutor(context)
            executor.add_node(EvaluateNode)

            inputs = {
                "raw_responses": str(raw_path),
                "eval_type": eval_type,
            }
            executor.run(inputs)


def cmd_pipeline(args):
    """Run full benchmark pipeline end-to-end."""
    config = _load_config(args)
    repeats = args.repeats if args.repeats is not None else config.runs.num_repeats

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
            subtasks = _select_subtasks(dim_name, dim_config.subtasks, args.subtask)
            for subtask in subtasks:
                for model_name in models:
                    print(f"\n[{dim_name}] [{subtask}] [{model_name}]")

                    executor = PipelineExecutor(context)
                    executor.add_node(LoadDatasetNode)
                    executor.add_node(BuildPromptsNode)
                    executor.add_node(RunInferenceNode)
                    executor.add_node(EvaluateNode)

                    eval_type = dim_config.eval_type

                    initial = {
                        "dimension": dim_name,
                        "subtask": subtask,
                        "model_name": model_name,
                        "vision": True,
                        "eval_type": eval_type,
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
    _load_config(args)
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
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    p_inspect = subparsers.add_parser(
        "inspect-data", help="Validate released questions and image files"
    )
    p_inspect.add_argument("--dimension", nargs="+", help="Specific dimensions to inspect")
    p_inspect.add_argument("--subtask", help="Specific subtask to inspect")
    p_inspect.add_argument(
        "--metadata-only",
        action="store_true",
        help="Validate question metadata without checking downloaded image files",
    )

    # inference command
    p_inference = subparsers.add_parser("inference", help="Run model inference")
    p_inference.add_argument("--model", nargs="+", help="Specific models to run")
    p_inference.add_argument("--dimension", nargs="+", help="Specific dimensions to run")
    p_inference.add_argument("--subtask", help="Specific configured subtask to run")
    p_inference.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of questions to load per selected subtask",
    )
    p_inference.add_argument("--run-id", default="run_0", help="Run identifier")
    p_inference.add_argument(
        "--text-only", action="store_true", help="Run without vision (text-only ablation)"
    )

    # evaluate command
    p_evaluate = subparsers.add_parser("evaluate", help="Evaluate model responses")
    p_evaluate.add_argument("--dimension", nargs="+", help="Specific dimensions to evaluate")
    p_evaluate.add_argument("--subtask", help="Specific configured subtask to evaluate")
    p_evaluate.add_argument("--model", nargs="+", help="Specific models to evaluate")
    p_evaluate.add_argument("--run-id", help="Specific run identifier to evaluate")

    # pipeline command
    p_pipeline = subparsers.add_parser("pipeline", help="Run full benchmark pipeline")
    p_pipeline.add_argument(
        "--repeats", type=int, help="Number of repeated runs (overrides config)"
    )
    p_pipeline.add_argument("--subtask", help="Specific configured subtask to run")
    p_pipeline.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of questions to load per selected subtask",
    )
    p_pipeline.add_argument(
        "--dry-run", action="store_true", help="Show pipeline graph without executing"
    )

    # report command
    p_report = subparsers.add_parser("report", help="Generate reports")
    p_report.add_argument("--output", "-o", default=".", help="Output directory for reports")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "inspect-data":
        cmd_inspect_data(args)
    elif args.command == "inference":
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
