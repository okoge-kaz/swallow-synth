import ast
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


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


def process_item_cpu(
    item: Dict[str, Any], input_target_key: str, output_target_key: str, tmp_dir: Path
) -> Dict[str, Any]:
    code: str = item.get(input_target_key, "")
    unique_path = tmp_dir / f"{uuid.uuid4()}.py"

    # format (includes syntax check)
    formatted, format_errors = format_with_ruff(code, unique_path)
    item[output_target_key] = formatted
    item["lint_report"] = format_errors

    return item
