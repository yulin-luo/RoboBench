"""RoboBench dataset loader.

Loads questions and metadata from the benchmark data directory.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class RoboBenchDataset:
    """Dataset loader for RoboBench benchmark data.

    Args:
        data_root: Root directory containing benchmark data
        middle_file_dir: Directory containing middle files (prompts ready for API)
    """

    def __init__(self, data_root: str = "", middle_file_dir: str = ""):
        self.data_root = Path(data_root) if data_root else Path()
        self.middle_dir = Path(middle_file_dir) if middle_file_dir else self.data_root / "middle_file"

    def load_questions(
        self,
        dimension: str,
        subtask: str = "",
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load questions for a given dimension/subtask.

        Searches the middle_file directory for matching JSONL files.
        """
        if not self.middle_dir.exists():
            print(f"Warning: middle_file directory not found: {self.middle_dir}")
            return []

        # Try to find matching file - exact naming pattern from RoboBench
        if subtask:
            patterns = [
                f"*{dimension}*{subtask}*questions.jsonl",
                f"*{subtask}*questions.jsonl",
                f"*{dimension}*.jsonl",
            ]
        else:
            patterns = [
                f"*{dimension}*questions.jsonl",
                f"*{dimension}*.jsonl",
            ]

        for pattern in patterns:
            matches = sorted(self.middle_dir.glob(pattern))
            if matches:
                print(f"  Loading: {matches[0].name} ({sum(1 for _ in open(matches[0]))} lines)")
                return self._load_jsonl(matches[0], max_samples=max_samples)

        # Fallback: search all jsonl files
        for f in sorted(self.middle_dir.iterdir()):
            dim_key = dimension.replace("_", "")
            file_key = f.name.replace("_", "")
            if f.suffix == ".jsonl" and (dim_key in file_key or subtask in f.name):
                print(f"  Loading (fallback): {f.name} ({sum(1 for _ in open(f))} lines)")
                return self._load_jsonl(f, max_samples=max_samples)

        print(f"  No matching middle file found for dimension='{dimension}' subtask='{subtask}'")
        return []

    def load_metadata(self, dimension: str, subtask: str = "") -> Dict[str, Any]:
        """Load metadata for a dimension."""
        # Placeholder: metadata can be extended as needed
        return {"dimension": dimension, "subtask": subtask}

    def _load_jsonl(
        self,
        path: Path,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load a JSONL file."""
        gt_index = self._load_gt_index(path)
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        self._attach_ground_truth(item, gt_index)
                        items.append(item)
                        if max_samples and len(items) >= max_samples:
                            break
                    except json.JSONDecodeError:
                        continue
        return items

    def _load_gt_index(self, middle_file: Path) -> Dict[str, Dict[str, Any]]:
        """Load released questions.json metadata matching a middle_file JSONL."""
        questions_path = self._infer_questions_json_path(middle_file)
        if not questions_path or not questions_path.exists():
            return {}

        try:
            with open(questions_path, "r", encoding="utf-8") as f:
                questions = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

        if not isinstance(questions, list):
            return {}
        return {
            q.get("unique_id", ""): q
            for q in questions
            if isinstance(q, dict) and q.get("unique_id")
        }

    def _infer_questions_json_path(self, middle_file: Path) -> Optional[Path]:
        """Map a prompt-ready middle_file name to the released questions.json."""
        stem = middle_file.name
        if stem.endswith("_questions.jsonl"):
            stem = stem[: -len("_questions.jsonl")]

        parts = stem.split("_")
        if parts and parts[0].isdigit():
            parts = parts[1:]
        if not parts:
            return None

        dimension_prefix = parts[0]
        dimension_map = {
            "instruction": "1_instruction_comprehension",
            "perception": "2_perception_reasoning",
            "generalized": "3_generalized_planning",
            "affordance": "4_affordance_reasoning",
            "error": "5_error_analysis",
        }
        dimension_dir = dimension_map.get(dimension_prefix)
        if not dimension_dir:
            return None

        subtask = Path(stem).name
        search_root = self.data_root / dimension_dir
        if not search_root.exists():
            return None

        candidates = []
        for qpath in search_root.rglob("questions.json"):
            rel_parts = qpath.relative_to(search_root).parts[:-1]
            rel_key = "_".join(rel_parts)
            if rel_key and rel_key in subtask:
                candidates.append((len(rel_key), qpath))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _attach_ground_truth(item: Dict[str, Any], gt_index: Dict[str, Dict[str, Any]]) -> None:
        """Attach gt fields from questions.json when the prompt JSONL omits them."""
        if item.get("gt_answer"):
            return

        request_id = item.get("request_id") or item.get("id")
        if not request_id:
            return

        base_id = request_id
        for suffix in ("_Q1", "_Q2", "_Q3"):
            if base_id.endswith(suffix):
                base_id = base_id[: -len(suffix)]
                break

        gt = gt_index.get(request_id) or gt_index.get(base_id)
        if not gt:
            return

        for key in ("gt_answer", "question_type", "options", "task_type", "input_type"):
            if key in gt and key not in item:
                item[key] = gt[key]
