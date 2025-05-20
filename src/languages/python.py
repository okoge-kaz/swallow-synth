import ast
import subprocess
from pathlib import Path
from typing import Dict, Any

from vllm import LLM, SamplingParams
from src.static.style_rewriting_prompt import PYTHON_STYLE_REWRITING_PROMPT, PYTHON_SELF_CONTAINED_REWRITING_PROMPT
# === CPU Stage: Syntax check, fast format (ruff), lint (ruff) ===


def syntax_check(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def format_with_ruff(code: str, tmp_path: Path) -> str:
    # Write code to temp file
    tmp_path.write_text(code, encoding="utf-8")
    try:
        # Ruff auto-fix (formatter)
        subprocess.run(["ruff", "format", str(tmp_path)], check=True, capture_output=True, text=True)
        return tmp_path.read_text(encoding="utf-8")
    except subprocess.CalledProcessError as e:
        # if format failed, return original code
        print(f"Ruff format failed: {e.stderr}")
        return code
    except Exception as e:
        # if other error (file I/O, etc.), return original code
        print(f"Unexpected error during formatting: {str(e)}")
        return code


def lint_with_ruff(code: str, tmp_path: Path) -> str:
    tmp_path.write_text(code, encoding="utf-8")
    # Ruff lint JSON output
    proc = subprocess.run(
        ["ruff", "--quiet", "--format=json", str(tmp_path)], capture_output=True, text=True, check=False
    )
    return proc.stdout


def process_item_cpu(item: Dict[str, Any], key: str = "text") -> Dict[str, Any]:
    code: str = item.get(key, "")
    # 1) Initial syntax
    ok: bool = syntax_check(code)
    item["syntax_error"] = not ok
    # 2) Fast format
    formatted: str = format_with_ruff(code, Path("tmp_code.py"))
    item["text_formatted"] = formatted
    # 3) Lint
    lint_report: str = lint_with_ruff(formatted, Path("tmp_code.py"))
    item["lint_report"] = lint_report
    return item


# === GPU Stage: Style & Self-contained rewriting via local LLM (vLLM) + post-check ===


class RewritePipeline:
    def __init__(self, model_name: str = "qwen-3"):  # adjust model path
        # Load local Qwen model on GPU
        self.llm = LLM(model=model_name)

    def generate(self, prompt: str) -> str:
        # Greedy / deterministic inference
        params = SamplingParams(temperature=0)
        output = self.llm.generate(prompt, params).first()
        return output.text

    def rewrite_style(self, code: str, lint_report: str) -> str:
        prompt = PYTHON_STYLE_REWRITING_PROMPT.format(lint_report=lint_report)
        return self.generate(prompt)

    def rewrite_self_contained(self, code: str) -> str:
        prompt = PYTHON_SELF_CONTAINED_REWRITING_PROMPT.format(code=code)
        return self.generate(prompt)

    def post_check(self, code: str) -> bool:
        return syntax_check(code)

    def process_item_gpu(self, item: Dict[str, Any]) -> Dict[str, Any]:
        formatted = item.get("formatted", "")
        lint_report = item.get("lint_report", "")
        # 4) Style-based rewrite
        rew1 = self.rewrite_style(formatted, lint_report)
        # 5) Self-contained rewrite
        rew2 = self.rewrite_self_contained(rew1)
        # 6) Post syntax check
        post_ok = self.post_check(rew2)
        item.update({"rewritten_style": rew1, "rewritten_self_contained": rew2, "post_syntax_ok": post_ok})
        return item
