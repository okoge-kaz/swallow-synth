from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import uuid


def syntax_check(code: str) -> Tuple[bool, List[Dict[str, Any]]]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".go", delete=False, encoding="utf-8") as tmp_file:
        tmp_file.write(code)
        tmp_path = Path(tmp_file.name)

    try:
        proc = subprocess.run(
            ["go", "tool", "compile", str(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return True, []
        message = (proc.stderr or proc.stdout or "go tool compile failed").strip()
        return False, [{"type": "syntax_error", "message": message}]
    except FileNotFoundError:
        return True, []
    finally:
        tmp_path.unlink(missing_ok=True)
        compiled = tmp_path.with_suffix(".o")
        compiled.unlink(missing_ok=True)


def format_with_gofmt(code: str, tmp_path: Optional[Path] = None) -> Tuple[str, List[Dict[str, Any]]]:
    if tmp_path is None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".go", delete=False, encoding="utf-8") as tmp_file:
            tmp_file.write(code)
            path = Path(tmp_file.name)
    else:
        tmp_path.write_text(code, encoding="utf-8")
        path = tmp_path

    try:
        proc = subprocess.run(
            ["gofmt", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        formatted = proc.stdout if proc.stdout else path.read_text(encoding="utf-8")
        return formatted, []
    except FileNotFoundError:
        return code, []
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or "gofmt failed").strip()
        return code, [{"type": "format_error", "message": message}]
    finally:
        path.unlink(missing_ok=True)


def process_item_cpu(
    item: Dict[str, Any],
    input_target_key: str,
    output_target_key: str,
    tmp_dir: Path,
) -> Dict[str, Any]:
    code: str = item.get(input_target_key, "")

    syntax_ok, syntax_errors = syntax_check(code)
    if not syntax_ok:
        item[output_target_key] = code
        item["lint_report"] = syntax_errors
        return item

    unique_path = tmp_dir / f"{uuid.uuid4()}.go"
    formatted, format_errors = format_with_gofmt(code, unique_path)

    item[output_target_key] = formatted
    item["lint_report"] = format_errors
    return item
