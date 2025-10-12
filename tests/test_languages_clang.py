from pathlib import Path
import subprocess

import pytest

from src.languages import c as lang_c, cpp as lang_cpp, cuda as lang_cuda


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


def _mock_run_success(monkeypatch: pytest.MonkeyPatch, expected_suffix: str):
    def fake_run(cmd, capture_output, text, check):  # type: ignore[unused-argument]
        path = Path(cmd[-1])
        assert path.suffix == expected_suffix
        path.write_text("formatted", encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_c_format_with_clang(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _mock_run_success(monkeypatch, ".c")
    formatted, errors = lang_c.format_with_clang_format("int main(){}", tmp_path / "code.c")
    assert formatted == "formatted"
    assert errors == []


def test_c_format_missing_clang(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def missing_run(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", missing_run)
    formatted, errors = lang_c.format_with_clang_format("int main(){}", tmp_path / "code.c")
    assert formatted == "int main(){}"
    assert errors == []


def test_c_syntax_check_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, capture_output, text, check):  # type: ignore[unused-argument]
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, errors = lang_c.syntax_check("int main(){return 0;}")
    assert ok is True
    assert errors == []


def test_c_syntax_check_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, capture_output, text, check):  # type: ignore[unused-argument]
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="error")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, errors = lang_c.syntax_check("int main(")
    assert ok is False
    assert errors[0]["type"] == "syntax_error"


def test_cpp_process_item_cpu(monkeypatch: pytest.MonkeyPatch, tmp_dir: Path) -> None:
    def fake_syntax(*_args, **_kwargs):
        return True, []

    def fake_format(code, tmp_path):  # type: ignore[unused-argument]
        return code + " // formatted", []

    monkeypatch.setattr(lang_cpp, "syntax_check", fake_syntax)
    monkeypatch.setattr(lang_cpp, "format_with_clang_format", fake_format)

    item = {"code": "int main() {return 0;}"}
    result = lang_cpp.process_item_cpu(item, "code", "formatted", tmp_dir)
    assert result["formatted"].endswith("// formatted")
    assert result["lint_report"] == []


def test_cuda_syntax_check_tool_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_run(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", missing_run)
    ok, errors = lang_cuda.syntax_check("__global__ void f(){}")
    assert ok is True
    assert errors == []


def test_cuda_format_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def failing_run(*_args, **_kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=["clang-format"], stderr="boom")

    monkeypatch.setattr(subprocess, "run", failing_run)
    formatted, errors = lang_cuda.format_with_clang_format("__global__ void f(){}", tmp_path / "code.cu")
    assert formatted == "__global__ void f(){}"
    assert errors[0]["message"] == "boom"
