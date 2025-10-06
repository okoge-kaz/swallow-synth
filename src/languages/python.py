import ast
import asyncio
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Iterator, AsyncIterator

from transformers import AutoTokenizer
from src.prompts.python.stage2 import PYTHON_STAGE2_PROMPT
from src.prompts.python.stage5 import PYTHON_STAGE5_PROMPT
from src.prompts.python.stage8 import PYTHON_STAGE8_PROMPT
from src.languages.abc import RewritePipeline

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


# === CPU Stage: Syntax check, fast format (ruff), lint (ruff) ===


def syntax_check(code: str) -> Tuple[bool, List[Dict[str, Any]]]:
    try:
        ast.parse(code)
        return True, []
    except SyntaxError as e:
        return False, [{"type": "syntax_error", "message": str(e), "line": e.lineno, "offset": e.offset}]


def format_with_ruff(code: str, tmp_path: Optional[Path] = None) -> Tuple[str, List[Dict[str, Any]]]:
    if tmp_path is None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp_file:
            tmp_file.write(code)
            tmp_path = Path(tmp_file.name)
    else:
        tmp_path.write_text(code, encoding="utf-8")

    try:
        # Ruff auto-fix (formatter)
        subprocess.run(["ruff", "format", str(tmp_path)], check=True, capture_output=True, text=True)
        formatted_code = tmp_path.read_text(encoding="utf-8")
        return formatted_code, []
    except subprocess.CalledProcessError as e:
        # if format failed, return original code and error
        print(f"Ruff format failed: {e.stderr}")
        return code, [{"type": "syntax_error", "message": e.stderr}]
    except Exception as e:
        # if other error (file I/O, etc.), return original code
        print(f"Unexpected error during formatting: {str(e)}")
        return code, []
    finally:
        tmp_path.unlink(missing_ok=True)


def lint_with_ruff(code: str, tmp_path: Optional[Path] = None) -> str:
    if tmp_path is None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp_file:
            tmp_file.write(code)
            tmp_path = Path(tmp_file.name)
    else:
        tmp_path.write_text(code, encoding="utf-8")

    try:
        # Ruff lint JSON output
        proc = subprocess.run(
            ["ruff", "--quiet", "--format=json", str(tmp_path)], capture_output=True, text=True, check=False
        )
        return proc.stdout
    except Exception as e:
        print(f"Unexpected error during linting: {str(e)}")
        return ""
    finally:
        tmp_path.unlink(missing_ok=True)


def process_item_cpu(item: Dict[str, Any], key: str = "text", output_key: str = "text_formatted") -> Dict[str, Any]:
    code: str = item.get(key, "")
    unique_path = Path(f"tmp_code_{uuid.uuid4().hex[:8]}.py")

    # format (includes syntax check)
    formatted, format_errors = format_with_ruff(code, unique_path)
    item[output_key] = formatted
    item["lint_report"] = format_errors

    return item


# === GPU Stage: Style & Self-contained rewriting via local LLM (vLLM) + post-check ===


def make_list(end: int):
    out = [x for x in (1, 2, 4, 8) if x <= end]
    v = 16
    while v <= end:
        out.append(v)
        v += 8
    return out


class PythonRewritePipeline(RewritePipeline):
    def __init__(
        self, model_name: str = "qwen-3", tensor_parallel_size: int = 1, max_model_len: int = 40960, use_async=False
    ) -> None:
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len

        # sync and tensorrt-llm backend
        if backend == "vllm":
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                max_num_seqs=512,
                gpu_memory_utilization=0.95,
                max_model_len=max_model_len,
            )
        else:
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def fix_errors(self, codes: list[str], lint_reports: list[str]) -> list[str]:
        # Construct chat templates for batch processing
        prompts: list[str] = []
        for code, lint_report in zip(codes, lint_reports):
            prompt: str = PYTHON_STAGE2_PROMPT.format(
                lint_report=lint_report,
                code=code,
            )
            prompts.append(prompt)
        tokenized_prompts_len = [len(self.tokenizer.encode(prompt)) for prompt in prompts]
        max_len = max(tokenized_prompts_len)
        if max_len >= self.max_model_len:
            raise ValueError(
                f"Prompt length exceeds model limit: {max_len} >= {self.max_model_len}. "
                "Consider reducing the input size or using a smaller model."
            )

        return [
            output.outputs[0].text
            for output in self.llm.generate(
                prompts, SamplingParams(temperature=0, max_tokens=self.max_model_len - max_len)
            )
        ]

