import pytest

from robobench.evaluation.multi_choice import MultiChoiceEvaluator


def evaluator_config(api, normalize_with_gpt=False):
    return {
        "gpt_model": "judge-model",
        "normalize_with_gpt": normalize_with_gpt,
        "api": api,
    }


def test_structured_answer_tag_is_scored_without_answer_word_pollution(benchmark_config):
    results = [
        {
            "request_id": "sample-1",
            "response": "<think>Reasoning mentions A and B.</think><answer>C</answer>",
            "gt_answer": "C",
        }
    ]

    scores = MultiChoiceEvaluator().evaluate(
        results,
        evaluator_config(benchmark_config.evaluation.api),
    )

    assert scores["correct"] == 1
    assert scores["accuracy"] == 1.0


def test_empty_response_counts_as_incorrect(benchmark_config):
    results = [
        {
            "request_id": "sample-1",
            "response": None,
            "gt_answer": "D",
        }
    ]

    scores = MultiChoiceEvaluator().evaluate(
        results,
        evaluator_config(benchmark_config.evaluation.api),
    )

    assert scores["total_questions"] == 1
    assert scores["correct"] == 0


def test_ambiguous_response_requires_explicit_normalization(benchmark_config):
    results = [
        {
            "request_id": "sample-1",
            "response": "I considered A, but the final answer is C.",
            "gt_answer": "C",
        }
    ]

    with pytest.raises(ValueError, match="normalize_with_gpt=true"):
        MultiChoiceEvaluator().evaluate(
            results,
            evaluator_config(benchmark_config.evaluation.api),
        )
