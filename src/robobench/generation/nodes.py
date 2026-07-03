"""Data generation nodes for the RoboBench pipeline.

Each node is a reusable component that transforms data from one stage to the next.
Users can compose these nodes to build custom benchmark generation pipelines.
"""

import json
import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class GenerationNode(ABC):
    """Base class for data generation nodes."""

    name: str = "base"

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the generation step.

        Args:
            inputs: Dictionary of input data

        Returns:
            Dictionary of output data
        """
        raise NotImplementedError


class MergeDataNode(GenerationNode):
    """Merge raw data sources (CSV metadata + planning JSONs).

    Input: Directory with images/, CSV file, planning JSON files
    Output: merged_planning.json + matched_data.csv
    """

    name = "merge_data"

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge data from multiple sources.

        Args:
            inputs: {
                "data_dir": str,  # Directory containing raw data
                "csv_file": str,  # Path to CSV metadata
                "planning_dir": str,  # Directory with planning JSONs
            }

        Returns:
            {
                "merged_planning": str,  # Path to merged_planning.json
                "matched_data": str,  # Path to matched_data.csv
            }
        """
        # TODO: Implement merging logic from code/1_merge_all_info_planning.py
        raise NotImplementedError("MergeDataNode not yet implemented. See code/1_merge_all_info_planning.py for reference.")


class GenerateExplicitQuestionsNode(GenerationNode):
    """Generate explicit planning questions (Q1/Q2/Q3).

    Input: merged_planning.json + matched_data.csv + template_planning.json
    Output: questions.json (explicit goals)
    """

    name = "generate_explicit_questions"

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explicit planning questions.

        Args:
            inputs: {
                "merged_planning": str,  # Path to merged_planning.json
                "matched_data": str,  # Path to matched_data.csv
                "template_file": str,  # Path to template_planning.json
                "output_dir": str,
            }

        Returns:
            {"questions": str}  # Path to questions.json
        """
        # TODO: Implement from code/2_task_planning_to_test_format.py
        raise NotImplementedError("GenerateExplicitQuestionsNode not yet implemented.")


class GenerateImplicitQuestionsNode(GenerationNode):
    """Generate implicit planning questions using GPT.

    Input: merged_planning.json + images
    Output: implicit_instructions.json → questions.json (implicit goals)
    """

    name = "generate_implicit_questions"

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate implicit instruction questions.

        Args:
            inputs: {
                "merged_planning": str,
                "images_dir": str,
                "api_config": dict,  # API config for GPT-4o
            }

        Returns:
            {"questions": str}
        """
        # TODO: Implement from code/4_implicit_demand_goal_generation.py
        raise NotImplementedError("GenerateImplicitQuestionsNode not yet implemented.")


class GenerateRobotTypeQuestionsNode(GenerationNode):
    """Generate robot type classification questions.

    Input: matched_data.csv (per robot type) + template_qa.json
    Output: questions.json (robot type QA)
    """

    name = "generate_robot_type_questions"

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate robot type questions.

        Args:
            inputs: {
                "matched_data": str,
                "template_file": str,
            }

        Returns:
            {"questions": str}
        """
        # TODO: Implement from code/3.1_robot_type_question_generation.py
        raise NotImplementedError("GenerateRobotTypeQuestionsNode not yet implemented.")


class GenerateViewTypeQuestionsNode(GenerationNode):
    """Generate camera view type classification questions.

    Input: image folders (folder names encode perspective)
    Output: questions.json (view type QA)
    """

    name = "generate_view_type_questions"

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate view type questions.

        Args:
            inputs: {
                "images_dir": str,
                "template_file": str,
            }

        Returns:
            {"questions": str}
        """
        # TODO: Implement from code/5.1_generate_view_type_question.py
        raise NotImplementedError("GenerateViewTypeQuestionsNode not yet implemented.")


class TranslateToFunctionNode(GenerationNode):
    """Translate natural language steps to function calls.

    Input: questions.json + model responses
    Output: enriched planning data with function call annotations
    """

    name = "translate_to_function"

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Translate natural language to function calls.

        Args:
            inputs: {
                "questions": str,
                "api_config": dict,
            }

        Returns:
            {"output": str}
        """
        # TODO: Implement from code/6_planning_translate_language_to_function.py
        raise NotImplementedError("TranslateToFunctionNode not yet implemented.")


class GenerationPipeline:
    """Pipeline for generating benchmark data.

    Composes GenerationNodes into a dataflow pipeline.
    """

    def __init__(self, nodes: Optional[List[GenerationNode]] = None):
        self.nodes = nodes or []

    def add_node(self, node: GenerationNode) -> "GenerationPipeline":
        self.nodes.append(node)
        return self

    def run(self, initial_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all nodes in sequence."""
        data = initial_inputs.copy()
        for node in self.nodes:
            print(f"Running: {node.name}")
            try:
                outputs = node.run(data)
                data.update(outputs)
            except NotImplementedError as e:
                print(f"  Node not implemented: {e}")
                continue
        return data
