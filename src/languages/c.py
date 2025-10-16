from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import uuid


class NonUTF8Error(Exception):
    """Raised when a subprocess output or file is not valid UTF-8."""


def _strict_utf8(data: bytes) -> str:
    return data.decode("utf-8", errors="strict")


_ENV_UTF8 = {**os.environ, "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8"}


def syntax_check(code: str) -> Tuple[bool, List[Dict[str, Any]]]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False, encoding="utf-8") as tmp_file:
        tmp_file.write(code)
        tmp_path = Path(tmp_file.name)

    try:
        cmd = ["gcc", "-fsyntax-only", "-fno-diagnostics-color", str(tmp_path)]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=False,
            check=False,
            env=_ENV_UTF8,
        )
        if proc.returncode == 0:
            return True, []
        try:
            message = (_strict_utf8(proc.stderr) or _strict_utf8(proc.stdout) or "gcc syntax check failed").strip()
        except UnicodeDecodeError as e:
            raise NonUTF8Error(f"gcc output is not valid UTF-8: {e}") from e

        return False, [{"type": "syntax_error", "message": message}]
    except FileNotFoundError:
        # gcc is not available; skip syntax check but do not block the pipeline
        return False, []
    finally:
        tmp_path.unlink(missing_ok=True)


def format_with_clang_format(code: str, tmp_path: Optional[Path] = None) -> Tuple[str, List[Dict[str, Any]]]:
    if tmp_path is None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False, encoding="utf-8") as tmp_file:
            tmp_file.write(code)
            path = Path(tmp_file.name)
    else:
        tmp_path.write_text(code, encoding="utf-8")
        path = tmp_path

    try:
        proc = subprocess.run(
            [
                "clang-format",
                "-i",
                "--assume-filename=temp.c",
                "-style=LLVM",
                str(path),
            ],
            capture_output=True,
            text=False,
            check=True,
            env=_ENV_UTF8,
        )
        if proc.stderr:
            try:
                _ = _strict_utf8(proc.stderr)
            except UnicodeDecodeError as e:
                raise NonUTF8Error(f"clang-format stderr is not valid UTF-8: {e}") from e
        try:
            formatted = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise NonUTF8Error(f"formatted file is not valid UTF-8: {e}") from e

        return formatted, []

    except FileNotFoundError:
        return code, []
    except subprocess.CalledProcessError as exc:
        try:
            msg = (_strict_utf8(exc.stderr) or _strict_utf8(exc.stdout) or "clang-format failed").strip()
        except UnicodeDecodeError as e:
            raise NonUTF8Error(f"clang-format error output is not valid UTF-8: {e}") from e
        return code, [{"type": "format_error", "message": msg}]
    finally:
        path.unlink(missing_ok=True)


def process_item_cpu(
    item: Dict[str, Any],
    input_target_key: str,
    output_target_key: str,
    tmp_dir: Path,
) -> Dict[str, Any]:
    code: str = item.get(input_target_key, "")

    try:
        syntax_ok, syntax_errors = syntax_check(code)
        if not syntax_ok:
            item[output_target_key] = code
            item["lint_report"] = syntax_errors
            return item

        unique_path = tmp_dir / f"{uuid.uuid4()}.c"
        formatted, format_errors = format_with_clang_format(code, unique_path)

        item[output_target_key] = formatted
        item["lint_report"] = format_errors
        return item

    except NonUTF8Error as e:
        item["drop"] = True
        item["lint_report"] = [{"type": "non_utf8", "message": str(e)}]
        return item
