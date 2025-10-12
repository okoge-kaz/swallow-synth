from pathlib import Path
import subprocess

import pytest

from src.languages import python as lang_py


def test_syntax_check_valid_code() -> None:
    ok, errors = lang_py.syntax_check("print('ok')\n")
    assert ok is True
    assert errors == []


def test_syntax_check_invalid_code() -> None:
    ok, errors = lang_py.syntax_check("def broken(:\n    pass\n")
    assert ok is False
    assert errors
    assert errors[0]["type"] == "syntax_error"


def test_format_with_ruff_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "snippet.py"

    def fake_run(cmd, check, capture_output, text):  # type: ignore[unused-argument]
        target.write_text("formatted", encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    formatted, errors = lang_py.format_with_ruff("original", target)
    assert formatted == "formatted"
    assert errors == []
    assert not target.exists()


def test_format_with_ruff_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "snippet.py"

    def failing_run(cmd, check, capture_output, text):  # type: ignore[unused-argument]
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="boom")

    monkeypatch.setattr(subprocess, "run", failing_run)

    formatted, errors = lang_py.format_with_ruff("source", target)
    assert formatted == "source"
    assert errors == [{"type": "syntax_error", "message": "boom"}]
    assert not target.exists()


def test_lint_with_ruff(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "lint.py"

    class Result:
        stdout = "lint output"

    def fake_run(cmd, capture_output, text, check):  # type: ignore[unused-argument]
        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)

    output = lang_py.lint_with_ruff("bad code", target)
    assert output == "lint output"
    assert not target.exists()


def test_process_item_cpu_uses_formatter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_format(code: str, tmp_path_arg: Path):
        assert tmp_path_arg.parent == tmp_path
        return code.upper(), [{"type": "syntax_error", "message": "issue"}]

    monkeypatch.setattr(lang_py, "format_with_ruff", fake_format)

    item = {"raw": "print('hi')"}
    result = lang_py.process_item_cpu(item, "raw", "formatted", tmp_path)

    assert result["formatted"] == "PRINT('HI')"
    assert result["lint_report"] == [{"type": "syntax_error", "message": "issue"}]
