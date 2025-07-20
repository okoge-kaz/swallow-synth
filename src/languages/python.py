import ast
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from src.prompts.python.stage2 import PYTHON_STAGE2_PROMPT
from src.prompts.python.stage4 import PYTHON_STAGE4_PROMPT
from src.prompts.python.stage5 import PYTHON_STAGE5_REWRITE_PROMPT
from src.prompts.python.competitive_programming import PYTHON_COMPETITIVE_PROGRAMMING_PROMPT
from src.languages.abc import RewritePipeline

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


def process_item_cpu(item: Dict[str, Any], key: str = "text") -> Dict[str, Any]:
    code: str = item.get(key, "")
    unique_path = Path(f"tmp_code_{uuid.uuid4().hex[:8]}.py")

    # format (includes syntax check)
    formatted, format_errors = format_with_ruff(code, unique_path)
    item["text_formatted"] = formatted
    item["lint_report"] = format_errors

    return item


# === GPU Stage: Style & Self-contained rewriting via local LLM (vLLM) + post-check ===


class PythonRewritePipeline(RewritePipeline):
    def __init__(self, model_name: str = "qwen-3", tensor_parallel_size: int = 1, max_model_len: int = 40960) -> None:
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.95,
            max_model_len=max_model_len,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    def rewrite_codes(self, codes: list[str]) -> list[str]:
        # Construct chat templates for batch processing
        prompts: list[str] = []

        PROMPT = PYTHON_STAGE5_REWRITE_PROMPT

        for code in codes:
            prompt = (
                "<|im_start|>system\n"
                + PROMPT
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

    def competitive_programming_write(self, questions: List[str]) -> List[str]:
        # Construct chat templates for batch processing
        prompts: List[str] = []

        PROMPT = PYTHON_COMPETITIVE_PROGRAMMING_PROMPT

        for question in questions:
            prompt = (
                "<|im_start|>system\n"
                + PROMPT
                + "<|im_end|>\n"
                + "<|im_start|>user\n"
                + f"{question}\n"
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
        return [output.outputs[0].text for output in outputs]
