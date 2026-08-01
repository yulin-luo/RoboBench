# -*- coding: utf-8 -*-
"""Async API client for model inference.

Consolidates the best features from all api_utils.py versions:
- Async concurrent calling with semaphore control
- Tenacity retry with exponential backoff
- Checkpoint / resume support
- Temp result saving every N completions
- Configurable timeout and concurrency
"""

import asyncio
from typing import Any, Dict, List, Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from .checkpoint import CheckpointManager


def _should_retry_api_error(exc: BaseException) -> bool:
    """Retry transient provider/API failures, but fail fast on bad payloads."""
    if isinstance(
        exc,
        (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError),
    ):
        return True
    if isinstance(exc, APIStatusError):
        status_code = exc.status_code
        return status_code == 429 or status_code >= 500
    return False


class AsyncModelClient:
    """Async client for calling MLLM APIs.

    Supports:
    - Configurable base_url, api_key, model
    - Semaphore-based concurrency control
    - Automatic retry with exponential backoff
    - Per-request timeout
    - Checkpoint/resume for long-running batches
    - Progress reporting
    """

    def __init__(self, api_config):
        """Initialize with an APIConfig object.

        Args:
            api_config: APIConfig instance from robobench.config
        """
        self.base_url = api_config.base_url
        self.api_key = api_config.api_key
        self.max_concurrent = api_config.api_max_concurrent
        self.retry_attempts = api_config.retry_attempts
        self.task_timeout = api_config.task_timeout
        self.extra_body = api_config.extra_body
        self.max_tokens = api_config.max_tokens
        backoff = api_config.retry_backoff
        self.backoff_multiplier = backoff.multiplier
        self.backoff_min = backoff.min
        self.backoff_max = backoff.max

    async def _call_api(
        self, messages: list, request_id: str, semaphore: asyncio.Semaphore, model: str
    ) -> Dict[str, Any]:
        """Single API call with retry."""
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(
                multiplier=self.backoff_multiplier,
                min=self.backoff_min,
                max=self.backoff_max,
            ),
            retry=retry_if_exception(_should_retry_api_error),
            reraise=True,
        )
        return await retrying(self._call_api_once, messages, request_id, semaphore, model)

    async def _call_api_once(
        self, messages: list, request_id: str, semaphore: asyncio.Semaphore, model: str
    ) -> Dict[str, Any]:
        """Execute one API attempt."""
        resp = {"id": request_id}
        # Provider-aware adaptation: Dubrify's OpenAI-compat layer does not translate
        # image_url -> image for Anthropic models, and Anthropic requires max_tokens.
        adapted_messages = self._adapt_messages_for_model(messages, model)
        kwargs = {"model": model, "messages": adapted_messages, "timeout": self.task_timeout}
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self._is_anthropic(model) and self.max_tokens is None:
            raise ValueError("api.max_tokens is required for Anthropic models")
        try:
            async with semaphore:
                async with AsyncOpenAI(
                    base_url=self.base_url, api_key=self.api_key, timeout=self.task_timeout
                ) as client:
                    # Start the per-request timer only after the task obtains a
                    # concurrency slot; otherwise low-concurrency batches can
                    # mark queued tasks as timed out before they make an API call.
                    response = await asyncio.wait_for(
                        client.chat.completions.create(**kwargs),
                        timeout=self.task_timeout,
                    )
                resp["response"] = response.choices[0].message.content
                resp["model"] = model
                return resp
        except Exception as e:
            print(f"Error in API call for request_id {request_id}: {type(e).__name__} - {str(e)}")
            raise

    @staticmethod
    def _is_anthropic(model: str) -> bool:
        m = model.lower()
        return m.startswith("claude") or m.startswith("anthropic/")

    @classmethod
    def _adapt_messages_for_model(cls, messages: list, model: str) -> list:
        """Convert OpenAI-style image content to provider-native format when needed.

        - OpenAI / Gemini: `{type: image_url, image_url: {url: data:image/...;base64,X}}`
        - Anthropic native: `{type: image, source: {type: base64, media_type: image/X, data: X}}`
        Dubrify forwards the body almost verbatim to Anthropic, which rejects the
        OpenAI-style tag, so we have to do the rewrite client-side.
        """
        if not cls._is_anthropic(model):
            return messages
        out = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                new_content = []
                for c in content:
                    if isinstance(c, dict) and c["type"] == "image_url":
                        url = c["image_url"]["url"]
                        if url.startswith("data:"):
                            header, b64 = url.split(",", 1)
                            media_type = header.split(":", 1)[1].split(";", 1)[0]
                            new_content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64,
                                    },
                                }
                            )
                        else:
                            # Public URL
                            new_content.append(
                                {
                                    "type": "image",
                                    "source": {"type": "url", "url": url},
                                }
                            )
                    else:
                        new_content.append(c)
                out.append({**msg, "content": new_content})
            else:
                out.append(msg)
        return out

    async def _wrapped_call(
        self,
        messages: list,
        request_id: str,
        index: int,
        model: str,
        semaphore: asyncio.Semaphore,
        results: list,
        completed_counter: list,
        failed_counter: list,
        save_path: str,
    ):
        """Wrap API call with timeout handling and progress tracking."""
        try:
            result = await self._call_api(messages, request_id, semaphore, model)
            completed_counter[0] += 1

            if completed_counter[0] % 100 == 0:
                print(f"Completed {completed_counter[0]} tasks")
                await self._save_temp_results([r for r in results if r is not None], save_path)

            return index, result

        except asyncio.TimeoutError:
            failed_counter[0] += 1
            print(f"Task {request_id} (index {index}) timed out after {self.task_timeout} seconds")
            return index, {
                "id": request_id,
                "response": None,
                "error": "timeout",
                "model": model,
            }

        except Exception as e:
            failed_counter[0] += 1
            print(
                f"Task {request_id} (index {index}) failed after all retries: "
                f"{type(e).__name__} - {str(e)}"
            )
            return index, {
                "id": request_id,
                "response": None,
                "error": str(e),
                "model": model,
            }

    async def _save_temp_results(self, results: list, save_path: str) -> None:
        """Save intermediate results to temp file."""
        import json as _json

        import aiofiles

        temp_file = f"{save_path}_temp_result.json"
        async with aiofiles.open(temp_file, "w", encoding="utf-8") as f:
            await f.write(_json.dumps(results, ensure_ascii=False, indent=2))

    async def run_batch(
        self,
        prompts: List[Dict[str, Any]],
        model: str,
        use_vision: bool = True,
        checkpoint: Optional[CheckpointManager] = None,
        save_path: str = "results",
    ) -> List[Dict[str, Any]]:
        """Run a batch of prompts against a model.

        Args:
            prompts: List of prompt dictionaries with 'messages' and 'request_id'
            model: Model name to call
            use_vision: Whether to include image content
            checkpoint: Optional checkpoint manager for resume
            save_path: Prefix for temp result files

        Returns:
            List of result dictionaries
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        results = [None] * len(prompts)
        existing_by_id = {}
        if checkpoint:
            existing_by_id = {
                result["id"]: result
                for result in checkpoint.get_existing_results()
                if result is not None and result["response"] is not None
            }
            if existing_by_id:
                print(f"Resuming with {len(existing_by_id)} completed requests")

        completed_count = [0]
        failed_count = [0]

        # Create tasks only for items not yet processed
        tasks = []
        for i, prompt in enumerate(prompts):
            request_id = prompt["request_id"]
            if request_id in existing_by_id:
                results[i] = existing_by_id[request_id]
                continue
            messages = prompt["messages"]

            # If not using vision, strip image content
            if not use_vision:
                messages = self._strip_vision(messages)

            task = self._wrapped_call(
                messages,
                request_id,
                i,
                model,
                semaphore,
                results,
                completed_count,
                failed_count,
                save_path,
            )
            tasks.append(task)

        print(f"Starting to process {len(tasks)} tasks, max concurrent: {self.max_concurrent}")

        # Execute tasks
        completed_tasks = 0
        for future in asyncio.as_completed(tasks):
            index, result = await future
            results[index] = result
            completed_tasks += 1

            if completed_tasks % 50 == 0 or completed_tasks == len(tasks):
                progress = completed_tasks / len(tasks) * 100 if tasks else 100
                print(
                    f"Progress: {completed_tasks}/{len(tasks)} "
                    f"({progress:.1f}%) - "
                    f"Success: {completed_count[0]}, Failed: {failed_count[0]}"
                )

                if checkpoint:
                    checkpoint.save(results)

        print(f"All tasks completed! Success: {completed_count[0]}, Failed: {failed_count[0]}")

        # Final save
        await self._save_temp_results(results, save_path)
        if checkpoint:
            checkpoint.save(results)

        return results

    @staticmethod
    def _strip_vision(messages: list) -> list:
        """Remove image content from messages for text-only ablation."""
        stripped = []
        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, list):
                    text_only = [c for c in content if c["type"] != "image_url"]
                    stripped.append({"role": "user", "content": text_only})
                else:
                    stripped.append(msg)
            else:
                stripped.append(msg)
        return stripped
