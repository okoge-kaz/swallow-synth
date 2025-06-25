import ast
import subprocess
import json
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from src.static.style_rewriting_prompt import PYTHON_STYLE_REWRITING_PROMPT
from src.prompts.python.stage2 import PYTHON_STAGE2_PROMPT
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

    def add_scores(self, codes: list[str]) -> list[str]:
        pass

    def rewrite_style(self, codes: list[str], lint_reports: list[str]) -> list[str]:
        # Construct chat templates for batch processing
        prompts: list[str] = []
        for code, lint_report in zip(codes, lint_reports):
            prompt = (
                "<|im_start|>system\n"
                "You are a code improvement assistant. Your task is to improve the given code based on the linting warnings and best practices.\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"{PYTHON_STYLE_REWRITING_PROMPT.format(lint_report=lint_report, code=code)}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            prompts.append(prompt)

        # Batch generate
        return self.generate(prompts)

    def process_batch(self, items: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """Process a batch of items through the pipeline."""
        # Extract formatted codes and lint reports
        formatted_codes = []
        lint_reports = []
        for item in items:
            formatted, _ = item.get("formatted", ("", []))
            formatted_codes.append(formatted)
            lint_report = item.get("lint_report", "")
            lint_reports.append(lint_report)

        # Filter and prioritize warnings for each item
        filtered_lint_reports = []
        for lint_report in lint_reports:
            lint_list = json.loads(lint_report) if lint_report else []
            filtered_lint_list = self.filter_and_prioritize_warnings(lint_list)
            filtered_lint_reports.append(json.dumps(filtered_lint_list))

        # Batch rewrite all codes
        rewritten_codes = self.rewrite_style(formatted_codes, filtered_lint_reports)

        # Update items with rewritten codes
        for item, rewritten in zip(items, rewritten_codes):
            item["rewritten_text"] = rewritten

        return items

    def process_item_gpu(self, items: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """Process items through GPU pipeline."""
        return self.process_batch(items)

    def filter_and_prioritize_warnings(self, lint_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and prioritize warnings to reduce noise and focus on important issues."""
        # Define warning priorities (higher number = higher priority)
        warning_priorities = {
            "syntax_error": 100,
            "undefined-variable": 70,
            "undefined-function": 70,
            "undefined-attribute": 70,
            "unused-variable": 60,
            "unused-import": 60,
            "unused-function": 60,
            "unused-class": 60,
            "redefined-outer-name": 50,
            "redefined-builtin": 50,
            "invalid-name": 40,
            "too-many-arguments": 20,
            "too-many-locals": 20,
            "too-many-statements": 20,
            "too-many-branches": 20,
            "too-many-return-statements": 20,
            "too-many-instance-attributes": 20,
            "too-few-public-methods": 20,
            "too-many-public-methods": 20,
            "too-many-ancestors": 20,
        }

        # Warnings that can be fixed by the prompt's guidelines
        prompt_fixable_warnings = {
            "missing-docstring",
            "missing-type-hint",
            "missing-return-type",
            "missing-param-type",
            "missing-param-doc",
            "missing-return-doc",
            "missing-raises-doc",
            "missing-yield-doc",
            "missing-yield-type",
            "missing-attribute-doc",
            "missing-class-doc",
            "missing-module-doc",
            "missing-function-doc",
            "missing-variable-doc",
            "missing-type-doc",
            "missing-any-doc",
            "missing-any-type",
            "missing-any-param",
            "missing-any-return",
            "missing-any-raises",
            "missing-any-yield",
            "missing-any-attribute",
            "missing-any-class",
            "missing-any-module",
            "missing-any-function",
            "missing-any-variable",
        }

        # Warnings that can be automatically fixed by ruff format
        format_fixable_warnings = {
            "line-too-long",
            "too-many-lines",
            "trailing-whitespace",
            "blank-line",
            "blank-lines",
            "format",
            "style",
            "whitespace",
            "indent",
            "indentation",
            "bad-indentation",
            "bad-continuation",
            "bad-line-continuation",
            "bad-line-break",
            "bad-line-wrap",
            "bad-line-split",
            "bad-line-join",
            "bad-line-spacing",
            "bad-line-indent",
            "bad-line-align",
            "bad-line-padding",
            "bad-line-margin",
            "bad-line-gap",
            "bad-line-space",
            "bad-line-tab",
            "bad-line-mix",
        }

        # Filter out low priority warnings, format-related warnings, and prompt-fixable warnings
        filtered_warnings = [
            warning
            for warning in lint_list
            if warning.get("type") in warning_priorities
            and warning_priorities[warning.get("type", "")] > 0
            and warning.get("type") not in prompt_fixable_warnings
            and warning.get("type") not in format_fixable_warnings
            and not warning.get("type", "").startswith("import-")  # Ignore all import-related warnings
        ]

        # Sort warnings by priority
        filtered_warnings.sort(key=lambda x: warning_priorities.get(x.get("type", ""), 0), reverse=True)

        # Limit the number of warnings to prevent overwhelming the LLM
        max_warnings = 10
        if len(filtered_warnings) > max_warnings:
            # Keep the highest priority warnings
            filtered_warnings = filtered_warnings[:max_warnings]
            # Add a note about additional warnings
            filtered_warnings.append(
                {
                    "type": "note",
                    "message": f"... and {len(lint_list) - max_warnings} more warnings (truncated for clarity)",
                }
            )

        return filtered_warnings
