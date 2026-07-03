"""Pipeline execution engine.

Orchestrates the dataflow graph of PipelineNodes, handling
topological ordering, checkpointing, and error recovery.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .context import RunContext
from .node import PipelineNode


class PipelineExecutor:
    """Executes a sequence of pipeline nodes."""

    def __init__(self, context: RunContext):
        self.context = context
        self.nodes: List[Type[PipelineNode]] = []
        self.results: Dict[str, Any] = {}

    def add_node(self, node_class: Type[PipelineNode]) -> "PipelineExecutor":
        """Add a node to the pipeline."""
        self.nodes.append(node_class)
        return self

    def run(self, initial_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute all nodes in sequence.

        Args:
            initial_inputs: Initial input data for the first node.

        Returns:
            Final outputs from the last node.
        """
        inputs = initial_inputs or {}
        start_time = time.time()

        print(f"{'=' * 60}")
        print(f"Pipeline started: run_id={self.context.run_id}")
        print(f"Nodes: {[n.name for n in self.nodes]}")
        print(f"{'=' * 60}")

        for i, node_class in enumerate(self.nodes):
            node = node_class(self.context)
            print(f"\n[{i + 1}/{len(self.nodes)}] Running: {node.name}")
            node_start = time.time()

            try:
                node.setup()
                node.validate_inputs(inputs)
                outputs = node.run(inputs)
                node.teardown()
            except Exception as e:
                print(f"ERROR in node '{node.name}': {e}")
                raise

            elapsed = time.time() - node_start
            print(f"  Completed in {elapsed:.1f}s")

            # Merge outputs into inputs for next node
            inputs.update(outputs)
            self.results[node.name] = outputs

        total_elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"Pipeline completed in {total_elapsed:.1f}s")
        print(f"{'=' * 60}")

        return inputs

    def run_stage(self, stage_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single stage by name."""
        for node_class in self.nodes:
            if node_class.name == stage_name:
                node = node_class(self.context)
                node.setup()
                outputs = node.run(inputs)
                node.teardown()
                return outputs
        raise ValueError(f"Stage '{stage_name}' not found in pipeline")

    def dry_run(self) -> List[str]:
        """Return the list of nodes that would be executed."""
        return [n.name for n in self.nodes]
