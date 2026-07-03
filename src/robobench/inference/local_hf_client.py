# -*- coding: utf-8 -*-
"""Local HuggingFace MLLM inference client for RoboBench.

Loads vision-language models directly from HuggingFace and runs generation
locally. Designed to produce the same output format as AsyncModelClient so that
scripts/run_e4_inference.py and downstream evaluators can consume the results
without modification.

Supported model families:
- Qwen3-VL (e.g. Qwen/Qwen3-VL-8B-Instruct, Alibaba-DAMO-Academy/RynnBrain-8B)
- Qwen2.5-VL (e.g. BAAI/RoboBrain2.0-7B, XiaomiMiMo/MiMo-Embodied-7B)
"""

import asyncio
import base64
import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# We import torch/transformers lazily so that this module can be imported in
# environments where only the API client is needed.

def _import_torch():
    import torch
    return torch


def _import_transformers():
    import transformers
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoModelClass
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelClass
    return transformers, AutoProcessor, AutoModelClass


def _decode_base64_image(url: str) -> Tuple[Image.Image, str]:
    """Parse a data:image/...;base64,XXX URL and return a PIL RGB image."""
    if not url.startswith("data:"):
        raise ValueError(f"Expected data URL, got {url[:60]}...")
    header, _, b64 = url.partition(",")
    media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"
    data = base64.b64decode(b64)
    img = Image.open(BytesIO(data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img, media_type


def _openai_to_qwen_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style messages (used by PromptBuilder) to Qwen format.

    Input may contain image_url entries with base64 data URLs. We decode them to
    PIL Images because Qwen processors accept both URLs and PIL images, and PIL
    images are more robust against processor-specific URL handling.
    """
    out = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        new_content = []
        for item in content:
            itype = item.get("type", "")
            if itype == "text":
                new_content.append({"type": "text", "text": item.get("text", "")})
            elif itype == "image_url":
                url = (item.get("image_url") or {}).get("url", "")
                if url.startswith("data:"):
                    img, _ = _decode_base64_image(url)
                    new_content.append({"type": "image", "image": img})
                elif url:
                    new_content.append({"type": "image", "image": url})
                else:
                    new_content.append({"type": "image", "image": ""})
            elif itype in ("image", "video"):
                # Already Qwen-style; keep as-is if it has a real image object.
                new_content.append(item)
            else:
                # Fallback: stringify unknown content parts.
                new_content.append({"type": "text", "text": str(item)})
        out.append({"role": role, "content": new_content})
    return out


class LocalHFClient:
    """Synchronous local HF model client.

    The constructor loads the model and processor. Calls to `generate` are
    synchronous (the model runs on GPU). For batch processing, use the async
    wrapper `LocalHFAsyncClient` below.
    """

    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        torch_dtype: Optional[str] = None,
        trust_remote_code: bool = False,
        token: Optional[str] = None,
        local_files_only: bool = False,
    ):
        torch = _import_torch()
        transformers, AutoProcessor, AutoModelClass = _import_transformers()

        self.model_id = model_id
        self.torch_dtype = getattr(torch, torch_dtype) if torch_dtype else torch.bfloat16
        self.token = token
        self.local_files_only = local_files_only

        print(f"[LocalHF] Loading processor for {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            token=token,
            local_files_only=local_files_only,
        )

        print(f"[LocalHF] Loading model {model_id} (dtype={self.torch_dtype}, device_map={device_map}) ...")
        self.model = AutoModelClass.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            token=token,
            local_files_only=local_files_only,
        )
        self.model.eval()
        print(f"[LocalHF] {model_id} ready.")

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> str:
        torch = _import_torch()

        qwen_messages = _openai_to_qwen_messages(messages)

        text = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # The processor expects a list of images (or image paths/urls) in the
        # same order as <|image_pad|> tokens in the prompt text.
        images = []
        for msg in qwen_messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") == "image":
                    images.append(item.get("image"))

        inputs = self.processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Strip the input prompt from generated IDs.
        prompt_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[:, prompt_len:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response

    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, Any]]],
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> List[str]:
        """Batch generation for multiple prompts.

        Runs synchronously on GPU. Much higher throughput than one-by-one
        because the model can process a batched prefill and shared vision
        encoder forward pass.
        """
        torch = _import_torch()

        qwen_messages_batch = [_openai_to_qwen_messages(msgs) for msgs in messages_batch]

        texts = []
        all_images: List[List[Any]] = []
        for qwen_messages in qwen_messages_batch:
            text = self.processor.apply_chat_template(
                qwen_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)

            images = []
            for msg in qwen_messages:
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                for item in content:
                    if item.get("type") == "image":
                        images.append(item.get("image"))
            all_images.append(images)

        # Build per-sample image lists. The processor handles padding and
        # interleaved image tokens as long as each sample's images are in a
        # separate sub-list.
        inputs = self.processor(
            text=texts,
            images=all_images if any(all_images) else None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_lens = inputs.input_ids.shape[1]
        generated_ids = output_ids[:, prompt_lens:]
        responses = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return responses


class LocalHFAsyncClient:
    """Async wrapper around LocalHFClient for compatibility with run_e4_inference.

    Runs the blocking `generate()` call in a thread pool so that multiple
    middle_files can be processed concurrently at the request level.
    """

    def __init__(self, model_id: str, batch_size: int = 4, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        self._batch_size = batch_size
        self._client: Optional[LocalHFClient] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor = None

    def _ensure_client(self):
        if self._client is None:
            self._client = LocalHFClient(self.model_id, **self.kwargs)
            self._loop = asyncio.get_running_loop()
            self._executor = None
        return self._client

    async def run_batch(
        self,
        prompts: List[Dict[str, Any]],
        model: str,
        use_vision: bool = True,
        checkpoint: Optional[Any] = None,
        save_path: str = "results",
        max_new_tokens: int = 512,
    ) -> List[Dict[str, Any]]:
        import json as _json

        client = self._ensure_client()

        # Resume from checkpoint if available
        if checkpoint:
            resume_idx = checkpoint.get_resume_index()
            existing = checkpoint.get_existing_results()
            if resume_idx > 0:
                print(f"[LocalHF] Resuming from index {resume_idx}")
                results = existing + [None] * (len(prompts) - len(existing))
            else:
                results = [None] * len(prompts)
        else:
            resume_idx = 0
            results = [None] * len(prompts)

        completed = 0
        failed = 0

        # Generate synchronously one-by-one. Empirically this is faster than
        # asyncio/thread-pool concurrency for transformers on a single GPU,
        # because it avoids kernel launch contention and gives each sample
        # the full GPU bandwidth.
        for i in range(resume_idx, len(prompts)):
            prompt = prompts[i]
            request_id = prompt.get("request_id", i)
            messages = prompt.get("messages", [])
            if not use_vision:
                messages = self._strip_vision(messages)
            try:
                response = client.generate(messages, max_new_tokens=max_new_tokens)
                results[i] = {"id": request_id, "response": response, "model": model}
                completed += 1
            except Exception as e:
                failed += 1
                print(f"[LocalHF] Error on request {request_id}: {type(e).__name__}: {e}")
                results[i] = {"id": request_id, "response": None, "error": str(e), "model": model}

            if (completed + failed) % 50 == 0 or (completed + failed) == len(prompts):
                print(
                    f"[LocalHF] Progress: {completed + failed}/{len(prompts)} "
                    f"(success={completed}, failed={failed})"
                )
                if checkpoint:
                    checkpoint.save(results, resume_idx + completed + failed - 1)

        return results

    @staticmethod
    def _strip_vision(messages: list) -> list:
        stripped = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    text_only = [c for c in content if c.get("type") != "image_url"]
                    stripped.append({"role": "user", "content": text_only})
                else:
                    stripped.append(msg)
            else:
                stripped.append(msg)
        return stripped
