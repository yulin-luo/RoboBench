"""RoboBench dataset loader."""

import json
from pathlib import Path
from typing import Any, Dict, List


class RoboBenchDataset:
    """Dataset loader for RoboBench benchmark data.

    Args:
        data_root: Root directory containing benchmark data
    """

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.system_prompts = self._load_json(self.data_root / "system_prompt.json")

    def load_questions(
        self,
        dimension: str,
        subtask: str,
        max_samples: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Load one configured subtask from its released questions.json."""
        dimension_dir = self._dimension_dir(dimension)
        matches = sorted((self.data_root / dimension_dir).rglob(f"{subtask}/questions.json"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one questions.json for dimension='{dimension}' "
                f"subtask='{subtask}', found {len(matches)} under {self.data_root / dimension_dir}"
            )

        questions = self._load_json(matches[0])
        if not isinstance(questions, list):
            raise TypeError(f"Expected a JSON list in {matches[0]}")
        print(f"  Loading: {matches[0].relative_to(self.data_root)} ({len(questions)} questions)")
        selected = questions[:max_samples] if max_samples is not None else questions
        return [self._prepare_question(question, matches[0].parent) for question in selected]

    @staticmethod
    def _load_json(path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _dimension_dir(self, dimension: str) -> str:
        dimension_map = {
            "instruction_comprehension": "1_instruction_comprehension",
            "perception_reasoning": "2_perception_reasoning",
            "generalized_planning": "3_generalized_planning",
            "affordance_reasoning": "4_affordance_reasoning",
            "error_analysis": "5_error_analysis",
        }
        return dimension_map[dimension]

    def _prepare_question(self, question: Dict[str, Any], benchmark_dir: Path) -> Dict[str, Any]:
        item = dict(question)
        item["request_id"] = question["unique_id"]
        item["prompt"] = self._build_prompt(question)
        item["image_urls"] = [
            str(self._resolve_image_path(question["unique_id"], image_name, benchmark_dir))
            for image_name in question["image_urls"]
        ]
        return item

    @staticmethod
    def _resolve_image_path(request_id: str, image_name: str, benchmark_dir: Path) -> Path:
        sample_id = request_id
        for suffix in ("_Q1", "_Q2", "_Q3"):
            if sample_id.endswith(suffix):
                sample_id = sample_id[: -len(suffix)]
                break
        sample_id = sample_id.split("/images/", 1)[1]
        image_root = benchmark_dir / "images"
        if Path(image_name).stem == Path(sample_id).name:
            return image_root / image_name
        image_path = image_root / sample_id / image_name
        return image_path

    def _build_prompt(self, question: Dict[str, Any]) -> str:
        dimension = question["task_type"][0]
        subtask = question["task_type"][-1]
        if dimension in {"1_instruction_comprehension", "3_generalized_planning"}:
            return self._build_planning_prompt(question, subtask)
        return self._build_multiple_choice_prompt(question, subtask)

    def _build_planning_prompt(self, question: Dict[str, Any], subtask: str) -> str:
        if subtask == "navigation":
            system_prompt = self.system_prompts["navigation_skill_list"]
        else:
            question_type = question["unique_id"][-2:]
            system_prompt = self.system_prompts[f"skill_list_{question_type}"]

        if subtask in {"color", "number", "shape", "size"}:
            system_prompt += self.system_prompts[subtask]

        robotic_type = (
            "single-arm" if question["robotic_type"] == "human" else question["robotic_type"]
        )
        context = f"You are a {robotic_type} robot."
        if subtask == "multi_view":
            view_names = [Path(image_name).stem for image_name in question["image_urls"]]
            context += (
                "The multiple images you have received now represent camera images from "
                "different views at the same time. The names of these views are "
                f"{view_names} in order."
            )
        return system_prompt + context + question["question"]

    def _build_multiple_choice_prompt(self, question: Dict[str, Any], subtask: str) -> str:
        system_prompt = ""
        if subtask == "static_attribute":
            attribute = question["unique_id"].split("/images/", 1)[1].split("/", 1)[0]
            attribute_prompts = {
                "is_sealed": "Sealed",
                "material": "Material",
                "contents": "Content",
                "can_contain_liquid": "Liquid",
                "transparency": "Transparency",
                "deformability": "Deformability",
                "fragility": "Fragility",
                "mass": "Mass",
            }
            system_prompt = self.system_prompts[attribute_prompts[attribute]]

        subtask_prompts = {
            "robot_type": "robot_type",
            "robot_view": "robot_view",
            "spatial_relation": "spatial_relation",
            "spatial_temporal_causality": "spatial_temporal_causality",
            "static_affordance": "afforadance_point",
            "dynamic_affordance": "afforadance_point",
            "navigation_visual_prompt": "afforadance_point",
            "high_level_planning_error": "high_level_planning_error",
            "low_level_execution_error": "low_level_execution_error",
        }
        if subtask in subtask_prompts:
            system_prompt = self.system_prompts[subtask_prompts[subtask]]

        options = "\n".join(
            f"{chr(ord('A') + index)}. {option}" for index, option in enumerate(question["options"])
        )
        return (
            system_prompt
            + question["question"]
            + "\n"
            + options
            + self.system_prompts["MCQ_post_prompt"]
        )
