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


@pytest.mark.parametrize(
    "language,stage,code_fence",
    [
        ("c", "stage4", "```c"),
        ("cpp", "stage4", "```cpp"),
        ("cuda", "stage4", "```cuda"),
        ("go", "stage4", "```go"),
        ("rust", "stage4", "```rust"),
        ("javascript", "stage4", "```javascript"),
        ("typescript", "stage4", "```typescript"),
    ],
)
def test_get_prompt_real_languages(language: str, stage: str, code_fence: str) -> None:
    prompt = get_prompt(stage, language)
    assert "<|REWRITTEN_CODE|>:" in prompt
    assert code_fence in prompt
