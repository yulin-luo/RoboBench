"""Base pipeline node and concrete implementations.

Each stage of the benchmark is a PipelineNode in a dataflow graph.
Nodes are connected by RunContext, which carries state between stages.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .context import RunContext


@dataclass
class NodeInput:
    """Definition of a node input."""

    name: str
    data_type: str = "any"  # "file", "directory", "memory"
    required: bool = True


@dataclass
class NodeOutput:
    """Definition of a node output."""

    name: str
    data_type: str = "any"


class PipelineNode(ABC):
    """Base class for all pipeline nodes.

    Subclasses must implement setup(), run(), and teardown().
    The run() method receives inputs and returns outputs as dictionaries.
    """

    name: str = "base"
    inputs: List[NodeInput] = field(default_factory=list)
    outputs: List[NodeOutput] = field(default_factory=list)

    def __init__(self, context: RunContext):
        self.context = context

    def setup(self) -> None:
        """Initialize the node. Called once before run()."""
        pass

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the node's logic.

        Args:
            inputs: Dictionary of input values keyed by input name.

        Returns:
            Dictionary of output values keyed by output name.
        """
        raise NotImplementedError

    def teardown(self) -> None:
        """Cleanup resources. Called once after run() completes."""
        pass

    def validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate that all required inputs are present."""
        for inp in self.inputs:
            if inp.required and inp.name not in inputs:
                raise ValueError(f"Node '{self.name}' missing required input: '{inp.name}'")


# ---------------------------------------------------------------------------
# Concrete Node Implementations
# ---------------------------------------------------------------------------


class LoadDatasetNode(PipelineNode):
    """Load questions for a given dimension/subtask."""

    name = "load_dataset"
    inputs = [NodeInput("dimension", "memory"), NodeInput("subtask", "memory")]
    outputs = [NodeOutput("questions", "memory")]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from robobench.data.dataset import RoboBenchDataset

        dimension = inputs["dimension"]
        subtask = inputs["subtask"]
        max_samples = inputs["max_samples"]
        paths = self.context.config.paths
        dataset = RoboBenchDataset(data_root=paths.data_root)
        questions = dataset.load_questions(dimension, subtask, max_samples=max_samples)
        return {"questions": questions}


class BuildPromptsNode(PipelineNode):
    """Construct API-ready prompts from questions."""

    name = "build_prompts"
    inputs = [NodeInput("questions", "memory")]
    outputs = [NodeOutput("prompts", "file"), NodeOutput("prompt_count", "memory")]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from robobench.prompts.builder import PromptBuilder

        questions = inputs["questions"]
        builder = PromptBuilder()
        prompts = builder.build(questions, mode="base64")
        output_path = self.context.get_temp_path("prompts.jsonl")
        builder.save(prompts, output_path)
        return {"prompts": str(output_path), "prompt_count": len(prompts)}


class RunInferenceNode(PipelineNode):
    """Call model API and save raw responses."""

    name = "run_inference"
    inputs = [
        NodeInput("prompts", "file"),
        NodeInput("model_name", "memory"),
        NodeInput("vision", "memory", required=False),
    ]
    outputs = [NodeOutput("raw_responses", "file"), NodeOutput("valid_count", "memory")]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import asyncio
        import json

        from robobench.inference.checkpoint import CheckpointManager
        from robobench.inference.client import AsyncModelClient

        prompts_path = inputs["prompts"]
        model_name = inputs["model_name"]
        use_vision = inputs["vision"]
        dimension = inputs["dimension"]
        subtask = inputs["subtask"]
        task_label = f"{dimension}_{subtask}"
        safe_task_label = task_label.replace("/", "_").replace("-", "_")
        safe_model_name = model_name.replace("/", "_").replace("-", "_")

        client = AsyncModelClient(self.context.config.api)
        checkpoint = CheckpointManager(
            self.context.get_temp_path(f"checkpoint_{safe_model_name}_{safe_task_label}.json")
        )

        # Load prompts from file
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = [json.loads(line) for line in f if line.strip()]

        # Run async inference
        results = asyncio.run(
            client.run_batch(
                prompts=prompts,
                model=model_name,
                use_vision=use_vision,
                checkpoint=checkpoint,
                save_path=str(
                    self.context.get_temp_path(f"temp_{safe_model_name}_{safe_task_label}")
                ),
            )
        )

        # Save results
        for idx, result in enumerate(results):
            if not isinstance(result, dict):
                raise TypeError(f"Expected inference result {idx} to be a dictionary")
            prompt = prompts[idx]
            raw_question = prompt["raw_question"]
            result["request_id"] = prompt["request_id"]
            result["raw_question"] = raw_question
            result["gt_answer"] = raw_question["gt_answer"]
            result["task_type"] = raw_question["task_type"]
            if "question_type" in raw_question:
                result["question_type"] = raw_question["question_type"]
            if "options" in raw_question:
                result["options"] = raw_question["options"]

        result_path = self.context.get_result_path(model_name, suffix=f"{safe_task_label}/raw.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        valid_count = sum(1 for result in results if result["response"])
        return {"raw_responses": str(result_path), "valid_count": valid_count}


class EvaluateNode(PipelineNode):
    """Run task-appropriate evaluator on raw responses."""

    name = "evaluate"
    inputs = [
        NodeInput("raw_responses", "file"),
        NodeInput("eval_type", "memory"),
    ]
    outputs = [NodeOutput("evaluated_results", "file"), NodeOutput("scores", "memory")]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import json

        from robobench.evaluation.base import get_evaluator

        eval_type = inputs["eval_type"]
        raw_path = inputs["raw_responses"]

        if eval_type == "planning":
            eval_config = {
                "eval_model": self.context.config.evaluation.planning.eval_model,
                "data_root": self.context.config.paths.data_root,
                "api": self.context.config.evaluation.api,
                "batch_save_prefix": str(Path(raw_path).parent / "judge"),
            }
        elif eval_type == "multi_choice":
            eval_config = {
                "gpt_model": self.context.config.evaluation.multi_choice.gpt_model,
                "normalize_with_gpt": (
                    self.context.config.evaluation.multi_choice.normalize_with_gpt
                ),
                "api": self.context.config.evaluation.api,
            }
        else:
            eval_config = {}

        with open(raw_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        evaluator = get_evaluator(eval_type)
        scores = evaluator.evaluate(results, eval_config)

        eval_path = Path(raw_path).parent / "evaluated.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)

        return {"evaluated_results": str(eval_path), "scores": scores}


class AggregateScoresNode(PipelineNode):
    """Aggregate scores across multiple runs."""

    name = "aggregate_scores"
    inputs = [NodeInput("run_scores", "memory")]  # List of score dicts
    outputs = [NodeOutput("aggregated", "memory"), NodeOutput("statistics", "file")]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import json

        from robobench.scoring.aggregator import RunAggregator

        run_scores = inputs["run_scores"]
        aggregator = RunAggregator()
        aggregated = aggregator.aggregate(run_scores)
        stats = aggregator.compute_statistics(aggregated)

        stats_path = Path(self.context.config.paths.results_root) / "aggregated" / "statistics.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(
                {"aggregated": aggregated, "statistics": stats},
                f,
                ensure_ascii=False,
                indent=2,
            )

        return {"aggregated": aggregated, "statistics": stats}
