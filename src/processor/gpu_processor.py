import asyncio
from itertools import count
import time
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Tuple, cast

from transformers import AutoTokenizer, PreTrainedTokenizer

from utils import apply_chat_template


try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    backend = "vllm"
except ImportError:
    try:
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

        backend = "tensorrt_llm"
    except ImportError as e:
        raise ImportError(e)
        backend = None

if backend is None:
    raise ImportError("Neither vllm nor tensorrt_llm is available.")
else:
    print(f"Using backend: {backend}")

from src.global_vars import get_logger


def make_list(end: int):
    out = [x for x in (1, 2, 4, 8) if x <= end]
    v = 16
    while v <= end:
        out.append(v)
        v += 8
    return out


class AsyncLLMClient:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        max_model_len: int,
        *,
        max_num_seqs: int = 512,  # tune per hardware
        gpu_memory_utilization: float = 0.95,
    ) -> None:
        self.logger = get_logger()
        self.logger.info(f"Initializing AsyncLLMClient with backend: {backend}")
        start_time = time.perf_counter()

        if backend == "vllm":
            engine_args = AsyncEngineArgs(
                model=model_name,
                max_num_seqs=max_num_seqs,
                task="generate",
                enable_prefix_caching=True,
                enforce_eager=True,
                async_scheduling=True,
                enable_chunked_prefill=True,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
            self.engine = AsyncLLM.from_engine_args(engine_args)
        elif backend == "tensorrt_llm":
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=1,
                max_seq_len=max_model_len,
                cuda_graph_config=CudaGraphConfig(
                    batch_sizes=make_list(512),
                    # batch_sizes=[1, 2, 4, 8, 16, 32, 48, 64, 128],
                    enable_padding=True,
                ),
                max_num_tokens=max_model_len * 8,  # TODO reduce when OOM
                max_batch_size=512,
                kv_cache_config=KvCacheConfig(
                    free_gpu_memory_fraction=0.9,
                    enable_block_reuse=True,
                ),
                enable_chunked_prefill=True,
            )

        self.logger.info(f"AsyncLLMClient initialized in {time.perf_counter() - start_time:.2f} seconds")

    async def generate(self, *, prompt: str, sampling_params: SamplingParams, request_id: str):
        """
        Minimal interface: given prompt + sampling params, return final text
        and token accounting from vLLM. No prompting or tokenization logic here.
        """
        if backend == "vllm":
            async for output in self.engine.generate(
                request_id=request_id, prompt=prompt, sampling_params=sampling_params
            ):
                if output.finished:
                    in_toks = len(output.prompt_token_ids or [])
                    out_toks = sum(len(stp.token_ids) for stp in output.outputs)
                    text = output.outputs[0].text if output.outputs else ""
                    return text, in_toks, out_toks
        elif backend == "tensorrt_llm":
            output = await self.llm.generate_async(prompt, sampling_params)
            text = output.outputs[0].text if output.outputs else ""
            in_toks = len(output.prompt_token_ids or [])
            out_toks = sum(len(stp.token_ids) for stp in output.outputs)
            return text, in_toks, out_toks


def fix_errors_processor_stage2(
    item: Dict[str, Any],
    *,
    tokenizer,
    max_model_len: int,
    template: str = "",
    code_key: str = "code",
    lint_key: str = "lint_report",
    temperature: float = 0.0,
) -> Tuple[str, SamplingParams]:
    if code_key not in item:
        raise KeyError(f"Item missing '{code_key}'.")
    if lint_key not in item:
        raise KeyError(f"Item missing '{lint_key}'.")

    code = item[code_key]
    lint_report = item[lint_key]

    if isinstance(lint_report, list):
        import json

        lint_report_str = json.dumps(lint_report, ensure_ascii=False)
    else:
        lint_report_str = str(lint_report)

    prompt = template.format(lint_report=lint_report_str, code=code)

    used = len(tokenizer.encode(prompt))
    if used >= max_model_len:
        raise ValueError(
            f"Prompt length exceeds model limit: {used} >= {max_model_len}. "
            "Consider truncating the input or increasing context."
        )
    max_tokens = max(1, max_model_len - used)

    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    return prompt, sp


def llm_rewrite_processor(
    item: Dict[str, Any],
    *,
    tokenizer,
    max_model_len: int,
    input_target_key: str,  # adjust with partial
    system_prompt: str = "",  # adjust with partial
    temperature: float = 0.0,  # adjust with partial
) -> Tuple[str, SamplingParams]:
    if input_target_key not in item:
        raise KeyError(f"Item missing required key '{input_target_key}'.")

    if system_prompt == "":
        raise ValueError("system_prompt is empty, use functools.partial to set a system prompt")

    code: str = item.get(input_target_key, "")
    prompt = apply_chat_template(tokenizer=tokenizer, system_prompt=system_prompt, user_input=code)

    used = len(tokenizer.encode(prompt))
    if used >= max_model_len:
        raise ValueError(
            f"Prompt length exceeds model limit: {used} >= {max_model_len}. "
            "Consider truncating input or increasing context."
        )
    max_tokens = max(1, max_model_len - used)

    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    return prompt, sp


def score_processor(
    item: Dict[str, Any],
    *,
    tokenizer: PreTrainedTokenizer,
    max_model_len: int,
    input_target_key: str,  # adjust with partial
    system_prompt: str,  # adjust with partial
    temperature: float = 0.0,  # adjust with partial
) -> Tuple[str, SamplingParams]:
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

    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    return prompt, sp


class CodeProcessor:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        max_model_len: int,
        use_async: bool = True,
    ) -> None:
        if not use_async:
            raise RuntimeError("This refactor is async-only. Provide a sync client if needed.")
        self.logger = get_logger()

        self.llm = AsyncLLMClient(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            max_num_seqs=512,
            gpu_memory_utilization=0.95,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = cast(PreTrainedTokenizer, self.tokenizer)
        self.max_model_len = max_model_len

    async def process_code(
        self,
        code_iterator: Iterator[Dict[str, Any]],
        *,
        processor: Callable[..., Tuple[str, SamplingParams]],
        max_in_flight: int = 2048,
    ) -> AsyncIterator[Dict[str, Any]]:
        rid_counter = count()
        pending: set[asyncio.Task] = set()

        totals = {"in": 0, "out": 0}
        start = time.perf_counter()

        async def make_task(item: Dict[str, Any]) -> asyncio.Task:
            rid = f"req-{next(rid_counter)}"
            prompt, sp = processor(item, tokenizer=self.tokenizer, max_model_len=self.max_model_len)

            async def _run():
                text, in_toks, out_toks = await self.llm.generate(prompt=prompt, sampling_params=sp, request_id=rid)
                return rid, item, text, in_toks, out_toks

            return asyncio.create_task(_run())

        # Prime
        for _ in range(max_in_flight):
            try:
                itm = next(code_iterator)
            except StopIteration:
                break
            pending.add(await make_task(itm))

        # Drain
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    rid, item, result, in_tokens, out_tokens = task.result()
                    totals["in"] += in_tokens
                    totals["out"] += out_tokens
                    elapsed = max(1e-6, time.perf_counter() - start)
                    self.logger.info(
                        f"{rid}, tokens/sec: in={totals['in'] / elapsed:.2f}, out={totals['out'] / elapsed:.2f}, elapsed: {elapsed:.2f}"
                    )
                    yield {"item": item, "result": result}
                except Exception as e:
                    self.logger.info(f"[task-error] {e!r}")
                    yield {"error": f"[swallow-code] [task-error] {e!r}"}

                try:
                    itm = next(code_iterator)
                except StopIteration:
                    itm = None
                if itm is not None:
                    pending.add(await make_task(itm))
