import ast
import asyncio
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Iterator, AsyncIterator

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from transformers import AutoTokenizer
from src.prompts.python.stage2 import PYTHON_STAGE2_PROMPT
from src.prompts.python.stage4 import PYTHON_STAGE4_PROMPT
from src.prompts.python.stage5 import PYTHON_STAGE5_REWRITE_PROMPT
from src.prompts.python.stage8 import PYTHON_STAGE8_REWRITE_PROMPT
from src.languages.abc import RewritePipeline


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


class PythonRewritePipeline(RewritePipeline):
    def __init__(self, model_name: str = "qwen-3", tensor_parallel_size: int = 1, max_model_len: int = 40960, use_async=False) -> None:
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len

        if use_async:
            engine_args = AsyncEngineArgs(
                model=model_name,
                max_num_seqs=512,  # H200 141GB SXM5 HBM3e x 8
                task="generate",
                enable_prefix_caching=True,
                enforce_eager=True,
                async_scheduling=True,
                enable_chunked_prefill=True,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=0.95,
                max_model_len=max_model_len,
            )
            self.engine = AsyncLLM.from_engine_args(engine_args)
        else:
            self.llm = LLM(
                model=model_name,
                max_num_seqs=512,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=0.95,
                max_model_len=max_model_len,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    async def run_request(self, engine, prompt, sampling_params, request_id):
        async for output in engine.generate(
            request_id=request_id, prompt=prompt, sampling_params=sampling_params
        ):
            if output.finished:
                # input tokens
                in_toks = len(output.prompt_token_ids or [])
                # output tokens
                out_toks = sum(len(stp.token_ids) for stp in output.outputs)
                return request_id, output.outputs[0].text, in_toks, out_toks

    def generate(self, prompts: list[str]) -> list[str]:
        # Greedy / deterministic inference
        params = SamplingParams(temperature=0)
        outputs = self.llm.generate(prompts, params)
        return [output.text for output in outputs]  # type: ignore

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

    def get_scores(self, codes: list[str]) -> list[str]:
        prompts: list[str] = []
        for code in codes:
            prompt: str = (
                "<|im_start|>system\n"
                + PYTHON_STAGE4_PROMPT
                + "<|im_end|>\n"
                + "<|im_start|>user\n"
                + f"{code}\n"
                + "<|im_end|>\n"
                + "<|im_start|>assistant\n"
            )
            prompts.append(prompt)
        tokenized_prompts_len = [len(self.tokenizer.encode(prompt)) for prompt in prompts]
        max_len = max(tokenized_prompts_len)
        if max_len >= self.max_model_len:
            raise ValueError(
                f"Prompt length exceeds model limit: {max_len} >= {self.max_model_len}. "
                "Consider reducing the input size or using a smaller model."
            )

        outputs = self.llm.generate(prompts, SamplingParams(temperature=0, max_tokens=self.max_model_len - max_len))
        return [output.outputs[0].text for output in outputs]  # type: ignore

    async def rewrite_codes(self, code_iterator: Iterator[dict[str, Any]], prompt_type: str = "stage5") -> AsyncIterator[Dict[str, Any]]:
        # Construct chat templates for batch processing
        pending = set()
        produced = 0
        consumed = 0
        max_in_flight = 2048

        total_in = 0
        total_out = 0

        async def make_task(code: str, prompt_type) -> None | asyncio.Task:
            nonlocal produced
            rid = f"req-{produced}"
            produced += 1

            # Select prompt based on prompt_type
            if prompt_type == "stage5":
                PROMPT = PYTHON_STAGE5_REWRITE_PROMPT
            elif prompt_type == "stage8":
                PROMPT = PYTHON_STAGE8_REWRITE_PROMPT
            else:
                raise ValueError(f"Unsupported prompt_type: {prompt_type}. Supported types: 'stage5', 'stage8'")

            prompt = (
                "<|im_start|>system\n"
                + PROMPT
                + "<|im_end|>\n"
                + "<|im_start|>user\n"
                + f"{code}\n"
                + "<|im_end|>\n"
                + "<|im_start|>assistant\n"
            )

            max_len = len(self.tokenizer.encode(prompt))
            sampling_params = SamplingParams(temperature=0, max_tokens=self.max_model_len - max_len)

            return asyncio.create_task(self.run_request(self.engine, prompt, sampling_params, rid))

        start = time.perf_counter()

        # Prime the pipeline up to max_in_flight
        for _ in range(max_in_flight):
            try:
                item = next(code_iterator)
            except StopIteration:
                break
            if "text_formatted" not in item:
                raise ValueError("All items in the batch must contain 'text_formatted' key for rewriting")
            pending.add(await make_task(item.get("text_formatted", ""), prompt_type))

        # As tasks complete, schedule new ones until inputs exhausted
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                consumed += 1
                try:
                    request_id, result, in_tokens, out_tokens = task.result()
                    total_in += in_tokens
                    total_out += out_tokens
                    elapsed = time.perf_counter() - start
                    print(f"Output tokens/sec: {total_out/elapsed:.2f}")
                    yield {
                        "result": result
                    }
                except Exception as e:
                    print(f"[task-error] {e!r}")
                    yield {
                        "error": f"[swallow-code] [task-error] {e!r}"
                    }

                # Refill
                try:
                    item = next(code_iterator)
                except StopIteration:
                    item = None
                if item is not None:
                    if "text_formatted" not in item:
                        raise ValueError("All items in the batch must contain 'text_formatted' key for rewriting")
                    pending.add(await make_task(item.get("text_formatted", ""), prompt_type))
