from pathlib import Path
import subprocess

import pytest

from src.languages import javascript as lang_js, typescript as lang_ts


def test_js_syntax_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, capture_output, text, check):  # type: ignore[unused-argument]
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, errors = lang_js.syntax_check("function x() {")
    assert ok is False
    assert errors[0]["message"] == "boom"


def test_js_format_missing_prettier(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def missing_run(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", missing_run)
    formatted, errors = lang_js.format_with_prettier("const x=1;", tmp_path / "code.js")
    assert formatted == "const x=1;"
    assert errors == []


def test_ts_syntax_tool_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_run(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", missing_run)
    ok, errors = lang_ts.syntax_check("const x: number = 1;")
    assert ok is True
    assert errors == []


def test_ts_process_item(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_syntax(*_args, **_kwargs):
        return True, []

    def fake_format(code, parser="typescript", tmp_path=None):  # type: ignore[unused-argument]
        return code + "\n", []

    monkeypatch.setattr(lang_ts, "syntax_check", fake_syntax)
    monkeypatch.setattr(lang_ts, "format_with_prettier", fake_format)

    item = {"code": "const x: number = 1;"}
    result = lang_ts.process_item_cpu(item, "code", "formatted", tmp_path)
    assert result["formatted"].endswith("\n")
    assert result["lint_report"] == []
