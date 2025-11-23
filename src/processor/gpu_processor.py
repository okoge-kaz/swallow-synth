from __future__ import annotations

import asyncio
from itertools import count
import time
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Tuple, cast

from transformers import AutoTokenizer, PreTrainedTokenizer

from src.global_vars import get_logger
from src.processor.gpu_backends import load_backend
from src.utils import apply_chat_template


SamplingParamsType = Any


def llm_rewrite_processor(
    item: Dict[str, Any],
    *,
    tokenizer,
    max_model_len: int,
    input_target_key: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    reasoning_effort: str = "medium",
    sampling_params_cls: type | None = None,
    medical_data: bool = False,
) -> Tuple[str, SamplingParamsType]:
    if sampling_params_cls is None:
        raise ValueError("sampling_params_cls must be provided for llm_rewrite_processor")

    if input_target_key not in item:
        raise KeyError(f"Item missing required key '{input_target_key}'.")

    if not medical_data:
        conversation = item.get(input_target_key, "")
        assert isinstance(conversation, list)
        user_turn = conversation[0]
        assert isinstance(user_turn, dict)
        assert user_turn.get("role") == "user"
        content: str = user_turn.get("content", "")
    else:
        content = item[input_target_key]

    prompt = apply_chat_template(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        user_input=content,
        reasoning_effort=reasoning_effort,
    )

    used = len(tokenizer.encode(prompt))
    if used >= max_model_len:
        raise ValueError(
            f"Prompt length exceeds model limit: {used} >= {max_model_len}. "
            "Consider truncating input or increasing context."
        )
    max_tokens = max(1, max_model_len - used)

    sp = sampling_params_cls(temperature=temperature, max_tokens=max_tokens, skip_special_tokens=False)
    return prompt, sp


def score_processor(
    item: Dict[str, Any],
    *,
    tokenizer: PreTrainedTokenizer,
    max_model_len: int,
    input_target_key: str,
    system_prompt: str,
    temperature: float = 0.0,
    sampling_params_cls: type | None = None,
) -> Tuple[str, SamplingParamsType]:
    if sampling_params_cls is None:
        raise ValueError("sampling_params_cls must be provided for score_processor")

    if input_target_key not in item:
        raise KeyError(f"Item missing required key '{input_target_key}'.")

    if system_prompt == "":
        raise ValueError("system_prompt is empty, use functools.partial to set a system prompt")

    text: str = item[input_target_key]
    prompt = apply_chat_template(tokenizer=tokenizer, system_prompt=system_prompt, user_input=text)
    used = len(tokenizer.encode(prompt))
    if used >= max_model_len:
        raise ValueError(
            f"Prompt length exceeds model limit: {used} >= {max_model_len}. Reduce input or increase model context."
        )
    max_tokens = max(1, max_model_len - used)

    sp = sampling_params_cls(temperature=temperature, max_tokens=max_tokens)
    return prompt, sp


class Processor:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        max_model_len: int,
        *,
        backend: str,
        max_num_seqs: int,
        use_async: bool = True,
    ) -> None:
        if not use_async:
            raise RuntimeError("This refactor is async-only. Provide a sync client if needed.")

        self.logger = get_logger()
        self.logger.info(f"Processor: Loading model '{model_name}' with backend '{backend}'...")
        backend_module = load_backend(backend)
        self.logger.info(f"Processor: Backend module '{backend}' loaded.")

        self.backend_name = backend

        self.sampling_params_cls = None
        if backend == "vllm":
            from vllm import SamplingParams

            self.sampling_params_cls = SamplingParams
        elif backend == "tensorrt-llm":
            from tensorrt_llm import SamplingParams

            self.sampling_params_cls = SamplingParams
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self.logger.info("Processor: llm client initializing...")
        self.llm = backend_module.AsyncLLMClient(  # type: ignore
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=0.92,
            logger=self.logger,
        )
        self.logger.info("Processor: llm client initialized.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = cast(PreTrainedTokenizer, self.tokenizer)
        self.max_model_len = max_model_len

    async def process_code(
        self,
        code_iterator: Iterator[Dict[str, Any]],
        *,
        processor: Callable[..., Tuple[str, SamplingParamsType]],
        max_in_flight: int = 2048,
    ) -> AsyncIterator[Dict[str, Any]]:
        rid_counter = count()
        pending: set[asyncio.Task] = set()

        totals = {"in": 0, "out": 0}
        start = time.perf_counter()

        async def make_task(item: Dict[str, Any]) -> asyncio.Task:
            rid = f"req-{next(rid_counter)}"
            prompt, sp = processor(
                item,
                tokenizer=self.tokenizer,
                max_model_len=self.max_model_len,
                sampling_params_cls=self.sampling_params_cls,
            )

            async def _run():
                text, in_toks, out_toks = await self.llm.generate(prompt=prompt, sampling_params=sp, request_id=rid)
                return rid, item, text, in_toks, out_toks

            return asyncio.create_task(_run())

        for _ in range(max_in_flight):
            try:
                itm = next(code_iterator)
            except StopIteration:
                break
            pending.add(await make_task(itm))

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    rid, item, result, in_tokens, out_tokens = task.result()
                    totals["in"] += in_tokens
                    totals["out"] += out_tokens
                    elapsed = max(1e-6, time.perf_counter() - start)
                    self.logger.info(
                        f"{rid}, backend={self.backend_name}, tokens/sec: in={totals['in'] / elapsed:.2f}, "
                        f"out={totals['out'] / elapsed:.2f}, elapsed: {elapsed:.2f}"
                    )
                    yield {"item": item, "result": result}
                except Exception as e:  # pragma: no cover - defensive logging
                    self.logger.info(f"[task-error] {e!r}")
                    yield {"error": f"[swallow-synth] [task-error] {e!r}"}

                try:
                    itm = next(code_iterator)
                except StopIteration:
                    itm = None
                if itm is not None:
                    pending.add(await make_task(itm))
