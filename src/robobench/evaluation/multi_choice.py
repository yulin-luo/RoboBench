"""Multiple-choice response evaluator."""

import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseEvaluator, register_evaluator


@register_evaluator("multi_choice")
class MultiChoiceEvaluator(BaseEvaluator):
    """Evaluate A-D answers with optional judge-based normalization."""

    name = "multi_choice"

    def __init__(self):
        self.judge_client = None

    def evaluate(
        self, results: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate all released MCQ responses."""
        if config is None:
            raise ValueError("Multiple-choice evaluation requires an explicit configuration")

        self.gpt_model = config["gpt_model"]
        self.normalize_with_gpt = config["normalize_with_gpt"]
        if self.normalize_with_gpt:
            from openai import OpenAI

            api = config["api"]
            self.judge_client = OpenAI(
                base_url=api.base_url,
                api_key=api.api_key,
                timeout=api.task_timeout,
            )

        correct = 0
        details = []
        for item in results:
            if not isinstance(item, dict):
                raise TypeError("Multiple-choice results must contain dictionaries")

            response = item["response"] or ""
            gt_answer = item["gt_answer"]
            score, explanation = self._evaluate_single(response, gt_answer)
            is_correct = score == 1.0
            correct += int(is_correct)
            details.append(
                {
                    "id": item["request_id"],
                    "question_type": "multiple_choice",
                    "score": score,
                    "is_correct": is_correct,
                    "explanation": explanation,
                }
            )

        total = len(results)
        accuracy = correct / total if total else 0
        return {
            "total_questions": total,
            "correct": correct,
            "accuracy": accuracy,
            "by_type": {"multiple_choice": accuracy},
            "details": details,
        }

    def _evaluate_single(self, response: str, gt_answer: str) -> Tuple[float, str]:
        response_letters = self._extract_response_letters(response)
        gt_letters = self._extract_option_letters(gt_answer)
        is_correct = response_letters == gt_letters
        explanation = (
            f"Response '{response_letters}' "
            f"{'matches' if is_correct else 'does not match'} GT '{gt_letters}'"
        )
        return (1.0 if is_correct else 0.0), explanation

    def _extract_response_letters(self, response: str) -> str:
        normalized = response.strip()
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            normalized,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if answer_match:
            normalized = answer_match.group(1).strip()
        else:
            normalized = re.sub(
                r"<think>.*?</think>",
                "",
                normalized,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()

        clean_letters = re.fullmatch(
            r"\s*([A-D])(?:\s*[,/&+]\s*([A-D]))*\s*[.)]?\s*",
            normalized,
            flags=re.IGNORECASE,
        )
        if clean_letters:
            return self._extract_option_letters(normalized)
        if not normalized:
            return ""
        if not self.normalize_with_gpt:
            raise ValueError(
                "Ambiguous multiple-choice response requires "
                "evaluation.multi_choice.normalize_with_gpt=true"
            )
        return self._normalize_with_gpt(normalized)

    @staticmethod
    def _extract_option_letters(text: str) -> str:
        return "".join(sorted(set(re.findall(r"[A-D]", text.upper()))))

    def _normalize_with_gpt(self, response: str) -> str:
        """Use the configured judge to extract the final A-D answer."""
        prompt = (
            "Extract only the final multiple-choice answer from this response.\n"
            "Return only A, B, C, or D.\n"
            f"Response: {response}"
        )
        completion = self.judge_client.chat.completions.create(
            model=self.gpt_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer = completion.choices[0].message.content.strip()
        if re.fullmatch(r"[A-D]", answer, flags=re.IGNORECASE) is None:
            raise ValueError(f"Judge returned an invalid multiple-choice answer: {answer!r}")
        return answer.upper()
