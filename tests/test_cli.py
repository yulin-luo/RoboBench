from types import SimpleNamespace

from robobench import cli


class RecordingExecutor:
    calls = []

    def __init__(self, context):
        self.context = context

    def add_node(self, node_class):
        return self

    def run(self, inputs):
        self.calls.append(inputs)
        return inputs


def inference_args():
    return SimpleNamespace(
        run_id="test-run",
        dimension=["perception_reasoning"],
        model=["test-model"],
        subtask=None,
        text_only=False,
        max_samples=None,
    )


def evaluate_args(tmp_path):
    return SimpleNamespace(
        dimension=["perception_reasoning"],
        subtask="tool_usage",
        model=["test-model"],
        run_id="test-run",
    )


def pipeline_args():
    return SimpleNamespace(
        repeats=1,
        dry_run=False,
        subtask=None,
        max_samples=None,
    )


def test_inference_iterates_all_configured_subtasks(monkeypatch, benchmark_config):
    RecordingExecutor.calls = []
    monkeypatch.setattr(cli, "_load_config", lambda args: benchmark_config)
    monkeypatch.setattr(cli, "PipelineExecutor", RecordingExecutor)

    cli.cmd_inference(inference_args())

    assert [call["subtask"] for call in RecordingExecutor.calls] == [
        "static_attribute",
        "tool_usage",
    ]


def test_pipeline_iterates_all_configured_subtasks(monkeypatch, benchmark_config):
    RecordingExecutor.calls = []
    monkeypatch.setattr(cli, "_load_config", lambda args: benchmark_config)
    monkeypatch.setattr(cli, "PipelineExecutor", RecordingExecutor)

    cli.cmd_pipeline(pipeline_args())

    assert [call["subtask"] for call in RecordingExecutor.calls] == [
        "static_attribute",
        "tool_usage",
    ]
    assert all(call["eval_type"] == "multi_choice" for call in RecordingExecutor.calls)


def test_evaluate_filters_to_one_subtask(monkeypatch, benchmark_config, tmp_path):
    RecordingExecutor.calls = []
    results_root = tmp_path / "results"
    benchmark_config.paths.results_root = str(results_root)
    for subtask in ("static_attribute", "tool_usage"):
        raw_path = (
            results_root
            / "test-run"
            / "test_model"
            / f"perception_reasoning_{subtask}"
            / "raw.json"
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(cli, "_load_config", lambda args: benchmark_config)
    monkeypatch.setattr(cli, "PipelineExecutor", RecordingExecutor)

    cli.cmd_evaluate(evaluate_args(tmp_path))

    assert len(RecordingExecutor.calls) == 1
    assert RecordingExecutor.calls[0]["raw_responses"].endswith(
        "perception_reasoning_tool_usage/raw.json"
    )
