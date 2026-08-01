"""Build API-ready messages from prepared RoboBench questions."""

import json
from pathlib import Path
from typing import Any, Dict, List


class PromptBuilder:
    """Build API messages from prepared RoboBench questions."""

    def build(
        self,
        questions: List[Dict[str, Any]],
        mode: str = "base64",
    ) -> List[Dict[str, Any]]:
        """Build a list of prompt messages from raw questions.

        Args:
            questions: List of raw question dicts
            mode: Image encoding mode ("base64" or "url")
        Returns:
            List of formatted prompt dicts with 'messages' and 'request_id'
        """
        if mode != "base64":
            raise ValueError("PromptBuilder only supports base64 image encoding")

        prompts = []
        for question in questions:
            prompt = self._build_single(question, mode)
            prompts.append(prompt)
        return prompts

    def _build_single(
        self,
        question: Dict[str, Any],
        mode: str,
    ) -> Dict[str, Any]:
        """Build a single prompt message."""
        request_id = question["request_id"]
        question_text = question["prompt"]
        image_urls = question["image_urls"]

        messages = []
        user_content = [{"type": "text", "text": question_text}]

        # Add images
        if image_urls and mode == "base64":
            from robobench.inference.image import process_image

            for img_path in image_urls:
                b64, media_type = process_image(img_path, resize=True, fmt="jpeg")
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64}"},
                    }
                )

        messages.append({"role": "user", "content": user_content})

        return {
            "request_id": request_id,
            "messages": messages,
            "raw_question": question,
        }

    @staticmethod
    def save(prompts: List[Dict[str, Any]], output_path: str | Path) -> None:
        """Save prompts to a JSONL file."""
        with open(output_path, "w", encoding="utf-8") as f:
            for prompt in prompts:
                f.write(json.dumps(prompt, ensure_ascii=False) + "\n")

    @staticmethod
    def load(input_path: str | Path) -> List[Dict[str, Any]]:
        """Load prompts from a JSONL file."""
        prompts = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(json.loads(line))
        return prompts
