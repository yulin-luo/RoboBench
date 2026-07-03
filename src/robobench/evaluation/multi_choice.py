"""Multi-choice, yes/no, and open-ended question evaluator.

Consolidates the best features from metric/multi-choice/evaluate_responses.py.
Supports:
- Multiple choice (A-D extraction with GPT normalization)
- Yes/No (direct string matching)
- Open-ended (GPT evaluation 0-1 or word overlap fallback)
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .base import BaseEvaluator, register_evaluator


@register_evaluator("multi_choice")
class MultiChoiceEvaluator(BaseEvaluator):
    """Evaluator for multiple choice, yes/no, and open-ended questions."""

    name = "multi_choice"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, gpt_model: str = "gpt-4o"):
        self.gpt_model = gpt_model
        self.openai_client = None
        if api_key:
            self.openai_client = OpenAI(base_url=base_url or "https://api.openai.com/v1", api_key=api_key)

    def evaluate(
        self, results: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate all responses in a results list.

        Args:
            results: List of result dicts with 'response' and 'gt_answer'
            config: Optional config with 'gpt_model', 'normalize_with_gpt'

        Returns:
            Dictionary with scores and statistics
        """
        if config:
            self.gpt_model = config.get("gpt_model", self.gpt_model)
            if not self.openai_client and config.get("normalize_with_gpt"):
                # Will use fallback methods
                pass

        total = 0
        correct = 0
        by_type = {"multiple_choice": {"total": 0, "correct": 0}, "yes_no": {"total": 0, "correct": 0}, "open_ended": {"total": 0, "correct": 0}}
        details = []

        for item in results:
            if not item or not isinstance(item, dict):
                continue

            response = item.get("response", "")
            gt_answer = item.get("gt_answer", item.get("ground_truth", ""))

            if not response or not gt_answer:
                continue

            total += 1
            qtype = self._determine_type(gt_answer)

            score, explanation = self._evaluate_single(response, gt_answer, qtype)
            is_correct = score >= 1.0 if qtype != "open_ended" else score >= 0.5

            if is_correct:
                correct += 1
                by_type[qtype]["correct"] += 1
            by_type[qtype]["total"] += 1

            details.append({
                "id": item.get("id", item.get("request_id", "")),
                "question_type": qtype,
                "score": score,
                "is_correct": is_correct,
                "explanation": explanation,
            })

        accuracy = correct / total if total > 0 else 0
        by_type_acc = {}
        for qtype, stats in by_type.items():
            by_type_acc[qtype] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

        return {
            "total_questions": total,
            "correct": correct,
            "accuracy": accuracy,
            "by_type": by_type_acc,
            "details": details,
        }

    def _determine_type(self, gt_answer: str) -> str:
        """Determine question type from ground truth."""
        if re.search(r"\b[A-D]\b", gt_answer.upper()):
            return "multiple_choice"

        yes_no_keywords = ["yes", "no", "equal", "left", "right", "unclear",
                           "transparent", "translucent", "opaque", "unknown"]
        if any(kw in gt_answer.lower() for kw in yes_no_keywords):
            return "yes_no"

        return "open_ended"

    def _evaluate_single(self, response: str, gt_answer: str, qtype: str) -> Tuple[float, str]:
        """Evaluate a single response."""
        if qtype == "multiple_choice":
            return self._eval_multiple_choice(response, gt_answer)
        elif qtype == "yes_no":
            return self._eval_yes_no(response, gt_answer)
        else:
            return self._eval_open_ended(response, gt_answer)

    def _eval_multiple_choice(self, response: str, gt: str) -> Tuple[float, str]:
        """Evaluate multiple choice answer."""
        # Extract letters from response
        resp_matches = re.findall(r"[A-D]", response.upper())
        gt_matches = re.findall(r"[A-D]", gt.upper())

        if len(resp_matches) > 1 and self.openai_client:
            # Use GPT to normalize ambiguous responses
            resp_matches = self._normalize_with_gpt(response)

        resp_normalized = "".join(sorted(set(resp_matches)))
        gt_normalized = "".join(sorted(set(gt_matches)))

        is_correct = resp_normalized == gt_normalized
        return (1.0 if is_correct else 0.0,
                f"Response '{resp_normalized}' {'matches' if is_correct else 'does not match'} GT '{gt_normalized}'")

    def _eval_yes_no(self, response: str, gt: str) -> Tuple[float, str]:
        """Evaluate yes/no answer."""
        resp_norm = response.lower().strip()
        gt_norm = gt.lower().strip()
        is_correct = resp_norm == gt_norm
        return (1.0 if is_correct else 0.0,
                f"Response '{resp_norm}' {'matches' if is_correct else 'does not match'} GT '{gt_norm}'")

    def _eval_open_ended(self, response: str, gt: str) -> Tuple[float, str]:
        """Evaluate open-ended answer."""
        if self.openai_client:
            return self._eval_with_gpt(response, gt)
        return self._eval_word_overlap(response, gt)

    def _normalize_with_gpt(self, response: str) -> List[str]:
        """Use GPT to extract the final answer from ambiguous responses."""
        prompt = (
            f"Extract ONLY the final multiple choice answer(s) from this response.\n"
            f"Return ONLY the letter(s) A,B,C,D with no other text.\n"
            f"If multiple answers, separate with commas.\n"
            f"Response: {response}"
        )
        try:
            completion = self.openai_client.chat.completions.create(
                model=self.gpt_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=80,
            )
            answer = completion.choices[0].message.content.strip()
            matches = re.findall(r"[A-D]", answer.upper())
            return matches
        except Exception as e:
            print(f"GPT normalization failed: {e}")
            return re.findall(r"[A-D]", response.upper())

    def _eval_with_gpt(self, response: str, gt: str) -> Tuple[float, str]:
        """Use GPT to evaluate open-ended answer (0-1 score)."""
        prompt = (
            f"Evaluate the accuracy of this response compared to the ground truth.\n\n"
            f"Ground truth: {gt}\n"
            f"Response: {response}\n\n"
            f"Score from 0 to 1:\n"
            f"- 1.0: Completely correct\n"
            f"- 0.5: Partially correct\n"
            f"- 0.0: Completely wrong\n\n"
            f"Return JSON: {{\"score\": float, \"explanation\": str}}"
        )
        try:
            completion = self.openai_client.chat.completions.create(
                model=self.gpt_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                timeout=80,
            )
            result = json.loads(completion.choices[0].message.content)
            return float(result.get("score", 0.0)), result.get("explanation", "")
        except Exception as e:
            print(f"GPT evaluation failed: {e}")
            return self._eval_word_overlap(response, gt)

    @staticmethod
    def _eval_word_overlap(response: str, gt: str) -> Tuple[float, str]:
        """Fallback: word overlap ratio scoring."""
        def normalize(text: str) -> set:
            return set(re.sub(r"[^\w\s]", "", text.lower()).split())

        resp_words = normalize(response)
        gt_words = normalize(gt)

        if not gt_words:
            return 0.0, "Ground truth is empty"

        overlap = len(resp_words & gt_words) / len(gt_words)

        if overlap >= 0.8:
            score = 1.0
        elif overlap >= 0.5:
            score = 0.75
        elif overlap >= 0.3:
            score = 0.5
        elif overlap > 0:
            score = 0.25
        else:
            score = 0.0

        return score, f"Word overlap: {overlap:.2f}"
