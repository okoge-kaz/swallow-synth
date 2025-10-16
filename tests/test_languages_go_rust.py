from pathlib import Path
import subprocess

import pytest

from src.languages import go as lang_go, rust as lang_rust


def test_go_format_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(cmd, capture_output, text, check):  # type: ignore[unused-argument]
        path = Path(cmd[-1])
        assert path.suffix == ".go"
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="package main\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    formatted, errors = lang_go.format_with_gofmt("package main", tmp_path / "code.go")
    assert formatted == "package main\n"
    assert errors == []


def test_go_syntax_tool_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_run(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", missing_run)
    ok, errors = lang_go.syntax_check("package main\nfunc main(){}")
    assert ok is True
    assert errors == []


def test_rust_format_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def failing_run(*_args, **_kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=["rustfmt"], stderr="boom")

    monkeypatch.setattr(subprocess, "run", failing_run)
    formatted, errors = lang_rust.format_with_rustfmt("fn main() {}", tmp_path / "code.rs")
    assert formatted == "fn main() {}"
    assert errors[0]["message"] == "boom"


def test_rust_process_item(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_syntax(*_args, **_kwargs):
        return True, []

    def fake_format(code, tmp_file):  # type: ignore[unused-argument]
        return code + "\n", []

    monkeypatch.setattr(lang_rust, "syntax_check", fake_syntax)
    monkeypatch.setattr(lang_rust, "format_with_rustfmt", fake_format)

    item = {"code": 'fn main() {println!("hi");}'}
    result = lang_rust.process_item_cpu(item, "code", "formatted", tmp_path)
    assert result["formatted"].endswith("\n")
    assert result["lint_report"] == []
