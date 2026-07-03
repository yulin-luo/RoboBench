"""Prompt builder for RoboBench.

Constructs API-ready prompts from raw question data.
Supports three modes:
  1. No-template: Directly use question text with optional system prompts
  2. Template: Use Python str.format() with question fields
  3. System prompt injection: Add system prompts based on question type

Reusable and open-source ready.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class PromptBuilder:
    """Builds prompts for robotic video analysis tasks.

    Args:
        data_root: Root directory for benchmark data
        system_prompt_key: Key to look up system prompts
        old_prefix: Old hardcoded path prefix to replace
    """

    def __init__(
        self,
        data_root: str = "",
        system_prompt_key: str = "",
        old_prefix: str = "/share/project/test/robobench/robobench/RoboBench-hf",
        new_prefix: str = "",
    ):
        self.data_root = data_root
        self.system_prompt_key = system_prompt_key
        self.old_prefix = old_prefix
        self.new_prefix = new_prefix or data_root

    def resolve_image_path(self, img_path: str) -> str:
        """Resolve image path by replacing old prefix with current data directory."""
        if self.old_prefix and img_path.startswith(self.old_prefix):
            return img_path.replace(self.old_prefix, self.new_prefix, 1)
        return img_path

    def build(
        self,
        questions: List[Dict[str, Any]],
        mode: str = "base64",
        system_prompts: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Build a list of prompt messages from raw questions.

        Args:
            questions: List of raw question dicts
            mode: Image encoding mode ("base64" or "url")
            system_prompts: Optional dict of system prompts keyed by type

        Returns:
            List of formatted prompt dicts with 'messages' and 'request_id'
        """
        prompts = []
        for question in questions:
            prompt = self._build_single(question, mode, system_prompts)
            prompts.append(prompt)
        return prompts

    def _build_single(
        self,
        question: Dict[str, Any],
        mode: str,
        system_prompts: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Build a single prompt message."""
        request_id = question.get("request_id", question.get("id", "unknown"))
        question_text = question.get("prompt", question.get("question", ""))
        instruction = question.get("instruction", "")
        image_urls = question.get("image_urls", [])

        # Build system prompt
        system_content = instruction
        if system_prompts and self.system_prompt_key:
            prefix = self._select_system_prompt(
                system_prompts, self.system_prompt_key, question
            )
            if prefix:
                system_content = prefix + (f"\n{instruction}" if instruction else "")

        # Build user content
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})

        user_content = [{"type": "text", "text": question_text}]

        # Add images
        if image_urls and mode == "base64":
            from robobench.inference.image import process_image

            for img_path in image_urls:
                resolved = self.resolve_image_path(img_path)
                try:
                    b64, media_type = process_image(resolved, resize=True, fmt="jpeg")
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{b64}"
                            },
                        }
                    )
                except FileNotFoundError:
                    # Skip missing images but log
                    print(f"Warning: Image not found: {resolved}")

        messages.append({"role": "user", "content": user_content})

        return {
            "request_id": request_id,
            "messages": messages,
            "raw_question": question,
        }

    def _select_system_prompt(
        self,
        system_prompts: Dict[str, str],
        key: str,
        question: Dict[str, Any],
    ) -> str:
        """Select appropriate system prompt based on question type."""
        # Direct lookup
        if key in system_prompts:
            return system_prompts[key]

        # Question-type-based selection
        question_type = str(question.get("type", "")).lower()
        for candidate in system_prompts:
            if candidate in question_type:
                return system_prompts[candidate]

        return ""

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
                    try:
                        prompts.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return prompts
