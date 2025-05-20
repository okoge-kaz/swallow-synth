import ast
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from vllm import LLM, SamplingParams
from src.static.style_rewriting_prompt import PYTHON_STYLE_REWRITING_PROMPT
# === CPU Stage: Syntax check, fast format (ruff), lint (ruff) ===


def syntax_check(code: str) -> Tuple[bool, List[Dict[str, Any]]]:
    try:
        ast.parse(code)
        return True, []
    except SyntaxError as e:
        return False, [{"type": "syntax_error", "message": str(e), "line": e.lineno, "offset": e.offset}]


def format_with_ruff(code: str, tmp_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    # Write code to temp file
    tmp_path.write_text(code, encoding="utf-8")
    try:
        # Ruff auto-fix (formatter)
        subprocess.run(["ruff", "format", str(tmp_path)], check=True, capture_output=True, text=True)
        return tmp_path.read_text(encoding="utf-8"), []
    except subprocess.CalledProcessError as e:
        # if format failed, return original code and error
        print(f"Ruff format failed: {e.stderr}")
        return code, [{"type": "syntax_error", "message": e.stderr}]
    except Exception as e:
        # if other error (file I/O, etc.), return original code
        print(f"Unexpected error during formatting: {str(e)}")
        return code, []


def lint_with_ruff(code: str, tmp_path: Path) -> str:
    tmp_path.write_text(code, encoding="utf-8")
    # Ruff lint JSON output
    proc = subprocess.run(
        ["ruff", "--quiet", "--format=json", str(tmp_path)], capture_output=True, text=True, check=False
    )
    return proc.stdout


def process_item_cpu(item: Dict[str, Any], key: str = "text") -> Dict[str, Any]:
    code: str = item.get(key, "")
    # 1) Fast format (includes syntax check)
    formatted, format_errors = format_with_ruff(code, Path("tmp_code.py"))
    item["text_formatted"] = formatted
    # 2) Lint
    lint_report: str = lint_with_ruff(formatted, Path("tmp_code.py"))

    # Add format errors to lint report
    if format_errors:
        # Convert existing lint report to list if it's not empty
        lint_list = json.loads(lint_report) if lint_report else []
        lint_list.extend(format_errors)
        lint_report = json.dumps(lint_list)

    item["lint_report"] = lint_report
    return item


# === GPU Stage: Style & Self-contained rewriting via local LLM (vLLM) + post-check ===


class RewritePipeline:
    def __init__(self, model_name: str = "qwen-3", tensor_parallel_size: int = 1):  # adjust model path
        # Load local Qwen model on GPU
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.95,
            max_model_len=131072,
        )

    def generate(self, prompt: str) -> str:
        # Greedy / deterministic inference
        params = SamplingParams(temperature=0)
        output = self.llm.generate(prompt, params)[0]
        return output.text  # type: ignore

    def rewrite_style(self, code: str, lint_report: str) -> str:
        prompt = PYTHON_STYLE_REWRITING_PROMPT.format(lint_report=lint_report, code=code)
        return self.generate(prompt)

    def post_check(self, code: str) -> Tuple[bool, List[Dict[str, Any]]]:
        return syntax_check(code)

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

    def process_item_gpu(self, item: Dict[str, Any]) -> Dict[str, Any]:
        formatted, _ = item.get("formatted", ("", []))
        lint_report = item.get("lint_report", "")

        # Filter and prioritize warnings
        lint_list = json.loads(lint_report) if lint_report else []
        filtered_lint_list = self.filter_and_prioritize_warnings(lint_list)
        filtered_lint_report = json.dumps(filtered_lint_list)

        # Single rewrite step that combines style and self-contained improvements
        rewritten = self.rewrite_style(formatted, filtered_lint_report)

        # Apply ruff format after rewriting
        rewritten, format_errors = format_with_ruff(rewritten, Path("tmp_code.py"))

        # Add format errors to lint report if any
        if format_errors:
            lint_list = json.loads(filtered_lint_report) if filtered_lint_report else []
            lint_list.extend(format_errors)
            filtered_lint_report = json.dumps(lint_list)
            # Try one more rewrite with error information
            rewritten = self.rewrite_style(rewritten, filtered_lint_report)
            # Apply ruff format again after the second rewrite
            rewritten, _ = format_with_ruff(rewritten, Path("tmp_code.py"))

        item.update({"rewritten_text": rewritten})
        return item
