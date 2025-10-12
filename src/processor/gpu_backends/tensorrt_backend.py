from __future__ import annotations

import time
from typing import Any, Tuple

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig


def _make_batch_sizes(limit: int) -> list[int]:
    sizes = [x for x in (1, 2, 4, 8) if x <= limit]
    current = 16
    while current <= limit:
        sizes.append(current)
        current += 8
    return sizes


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
        del max_num_seqs
        del gpu_memory_utilization
        self.logger = logger
        self.logger.info("Initializing AsyncLLMClient with backend: tensorrt-llm")
        start_time = time.perf_counter()

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=1,
            max_seq_len=max_model_len,
            cuda_graph_config=CudaGraphConfig(
                batch_sizes=_make_batch_sizes(512),
                enable_padding=True,
            ),
            max_num_tokens=max_model_len * 8,
            max_batch_size=512,
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=0.9,
                enable_block_reuse=True,
            ),
            enable_chunked_prefill=True,
        )

        elapsed = time.perf_counter() - start_time
        self.logger.info(f"AsyncLLMClient (tensorrt-llm) initialized in {elapsed:.2f} seconds")

    async def generate(self, *, prompt: str, sampling_params: SamplingParams, request_id: str) -> Tuple[str, int, int]:
        del request_id
        output = await self.llm.generate_async(prompt, sampling_params)
        text = output.outputs[0].text if output.outputs else ""
        in_toks = len(output.prompt_token_ids or [])
        out_toks = sum(len(stp.token_ids) for stp in output.outputs)
        return text, in_toks, out_toks
