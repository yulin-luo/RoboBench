import json
from pathlib import Path

import pytest

from robobench.data.dataset import RoboBenchDataset


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_system_prompts(root: Path) -> None:
    write_json(
        root / "system_prompt.json",
        {
            "Sealed": "SEALED\n",
            "MCQ_post_prompt": "\nPOST",
            "skill_list_Q1": "SKILLS\n",
        },
    )


def test_builds_prompts_and_image_paths_from_released_data(tmp_path):
    write_system_prompts(tmp_path)
    static_path = (
        tmp_path
        / "2_perception_reasoning"
        / "object_centric"
        / "static_attribute"
        / "questions.json"
    )
    write_json(
        static_path,
        [
            {
                "unique_id": (
                    "2_perception_reasoning/object_centric/static_attribute/"
                    "images/is_sealed/is_sealed_0000"
                ),
                "task_type": [
                    "2_perception_reasoning",
                    "object_centric",
                    "static_attribute",
                ],
                "question": "Is it sealed?",
                "gt_answer": "B",
                "image_urls": ["frame.png"],
                "options": ["No", "Yes"],
            }
        ],
    )
    planning_path = tmp_path / "1_instruction_comprehension" / "explicit_object_goal"
    write_json(
        planning_path / "questions.json",
        [
            {
                "unique_id": (
                    "1_instruction_comprehension/explicit_object_goal/images/episode_1_Q1"
                ),
                "task_type": ["1_instruction_comprehension", "explicit_object_goal"],
                "question": "Plan the task.",
                "gt_answer": "1-grasp(object)",
                "image_urls": ["frame_00.png"],
                "robotic_type": "human",
            }
        ],
    )

    dataset = RoboBenchDataset(str(tmp_path))
    static = dataset.load_questions("perception_reasoning", "static_attribute")[0]
    planning = dataset.load_questions("instruction_comprehension", "explicit_object_goal")[0]

    assert static["prompt"] == "SEALED\nIs it sealed?\nA. No\nB. Yes\nPOST"
    assert static["image_urls"] == [
        str(static_path.parent / "images" / "is_sealed" / "is_sealed_0000" / "frame.png")
    ]
    assert planning["prompt"] == "SKILLS\nYou are a single-arm robot.Plan the task."
    assert planning["image_urls"] == [str(planning_path / "images" / "episode_1" / "frame_00.png")]


def test_flat_image_layout_is_preserved(tmp_path):
    write_system_prompts(tmp_path)
    questions_path = (
        tmp_path / "2_perception_reasoning" / "object_centric" / "tool_usage" / "questions.json"
    )
    write_json(
        questions_path,
        [
            {
                "unique_id": ("2_perception_reasoning/object_centric/tool_usage/images/sample_1"),
                "task_type": ["2_perception_reasoning", "object_centric", "tool_usage"],
                "question": "What is it used for?",
                "gt_answer": "A",
                "image_urls": ["sample_1.png"],
                "options": ["Use A", "Use B"],
            }
        ],
    )

    question = RoboBenchDataset(str(tmp_path)).load_questions("perception_reasoning", "tool_usage")[
        0
    ]

    assert question["image_urls"] == [str(questions_path.parent / "images" / "sample_1.png")]


def test_subtask_lookup_requires_exactly_one_questions_file(tmp_path):
    write_system_prompts(tmp_path)
    dataset = RoboBenchDataset(str(tmp_path))

    with pytest.raises(FileNotFoundError, match="found 0"):
        dataset.load_questions("perception_reasoning", "static_attribute")

    write_json(
        tmp_path / "2_perception_reasoning" / "first" / "static_attribute" / "questions.json",
        [],
    )
    write_json(
        tmp_path / "2_perception_reasoning" / "second" / "static_attribute" / "questions.json",
        [],
    )

    with pytest.raises(FileNotFoundError, match="found 2"):
        dataset.load_questions("perception_reasoning", "static_attribute")
