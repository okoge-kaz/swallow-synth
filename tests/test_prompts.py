import pytest

from src.prompts import get_prompt


def test_get_prompt_returns_expected_prompt(monkeypatch) -> None:
    prompt_text = "example prompt"

    class DummyModule:
        PYTHON_STAGE3_PROMPT = prompt_text

    def fake_import(name: str):
        if name == "src.prompts.python.stage3":
            return DummyModule
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("src.prompts.import_module", fake_import)
    assert get_prompt("stage3", "python") == prompt_text


def test_get_prompt_invalid_language() -> None:
    with pytest.raises(ValueError):
        get_prompt("stage3", "go")


def test_get_prompt_missing_module(monkeypatch) -> None:
    def fake_import(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("src.prompts.import_module", fake_import)

    with pytest.raises(ValueError) as exc:
        get_prompt("stage99", "python")
    assert "Prompt module not found" in str(exc.value)
