import subprocess
from pathlib import Path
from typing import Dict, Any

from vllm import LLM, SamplingParams
from src.static.style_rewriting_prompt import C_STYLE_REWRITING_PROMPT, C_SELF_CONTAINED_REWRITING_PROMPT


def syntax_check(code: str) -> bool:
    try:
        # Write code to temp file
        tmp_path = Path("tmp_code.c")
        tmp_path.write_text(code, encoding="utf-8")
        # Check syntax with gcc
        result = subprocess.run(["gcc", "-fsyntax-only", str(tmp_path)], capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Syntax check failed: {str(e)}")
        return False


def format_with_clang_format(code: str, tmp_path: Path) -> str:
    # Write code to temp file
    tmp_path.write_text(code, encoding="utf-8")
    try:
        # Clang Format
        subprocess.run(["clang-format", "-i", str(tmp_path)], check=True, capture_output=True, text=True)
        return tmp_path.read_text(encoding="utf-8")
    except subprocess.CalledProcessError as e:
        print(f"Clang Format failed: {e.stderr}")
        return code
    except Exception as e:
        print(f"Unexpected error during formatting: {str(e)}")
        return code


def lint_with_cppcheck(code: str, tmp_path: Path) -> str:
    tmp_path.write_text(code, encoding="utf-8")
    try:
        # Cppcheck lint XML output
        proc = subprocess.run(
            ["cppcheck", "--xml", "--xml-version=2", str(tmp_path)], capture_output=True, text=True, check=False
        )
        return proc.stdout
    except Exception as e:
        print(f"Cppcheck lint failed: {str(e)}")
        return "[]"


def process_item_cpu(item: Dict[str, Any], key: str = "text") -> Dict[str, Any]:
    code: str = item.get(key, "")
    # 1) Initial syntax
    ok: bool = syntax_check(code)
    item["syntax_error"] = not ok
    # 2) Fast format
    formatted: str = format_with_clang_format(code, Path("tmp_code.c"))
    item["text_formatted"] = formatted
    # 3) Lint
    lint_report: str = lint_with_cppcheck(formatted, Path("tmp_code.c"))
    item["lint_report"] = lint_report
    return item


class RewritePipeline:
    def __init__(self, model_name: str = "qwen-3"):
        # Load local Qwen model on GPU
        self.llm = LLM(model=model_name)

    def generate(self, prompt: str) -> str:
        # Greedy / deterministic inference
        params = SamplingParams(temperature=0)
        output = self.llm.generate(prompt, params).first()
        return output.text

    def rewrite_style(self, code: str, lint_report: str) -> str:
        prompt = C_STYLE_REWRITING_PROMPT.format(lint_report=lint_report)
        return self.generate(prompt)

    def rewrite_self_contained(self, code: str) -> str:
        prompt = C_SELF_CONTAINED_REWRITING_PROMPT.format(code=code)
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
