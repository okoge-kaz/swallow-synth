from __future__ import annotations

import time
from typing import Any, Tuple

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM


class AsyncLLMClient:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        max_model_len: int,
        *,
        max_num_seqs: int = 512,
        gpu_memory_utilization: float = 0.95,
        logger: Any,
    ) -> None:
        self.logger = logger
        self.logger.info("Initializing AsyncLLMClient with backend: vllm")
        start_time = time.perf_counter()

        engine_args = AsyncEngineArgs(
            model=model_name,
            max_num_seqs=max_num_seqs,
            task="generate",
            enable_prefix_caching=True,
            enforce_eager=False,
            async_scheduling=True,
            enable_chunked_prefill=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            # compilation_config=
        )
        self.engine = AsyncLLM.from_engine_args(engine_args)

        elapsed = time.perf_counter() - start_time
        self.logger.info(f"AsyncLLMClient (vllm) initialized in {elapsed:.2f} seconds")

    async def generate(self, *, prompt: str, sampling_params: SamplingParams, request_id: str) -> Tuple[str, int, int]:
        async for output in self.engine.generate(request_id=request_id, prompt=prompt, sampling_params=sampling_params):
            if output.finished:
                in_toks = len(output.prompt_token_ids or [])
                out_toks = sum(len(stp.token_ids) for stp in output.outputs)
                text = output.outputs[0].text if output.outputs else ""
                return text, in_toks, out_toks
        return "", 0, 0
