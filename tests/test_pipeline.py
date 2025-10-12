import json
from pathlib import Path
import sys
import types

import pytest

from src.pipeline import stream_jsonl_


class _DummyProcessor:
    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - minimal stub
        pass

    async def process_code(self, *_args, **_kwargs):  # pragma: no cover - not used in these tests
        yield {}


def _dummy_processor_fn(*_args, **_kwargs):
    return "", None


stub_gpu = types.ModuleType("processor.gpu_processor")
stub_gpu.CodeProcessor = _DummyProcessor
stub_gpu.llm_rewrite_processor = _dummy_processor_fn
stub_gpu.score_processor = _dummy_processor_fn
sys.modules.setdefault("processor.gpu_processor", stub_gpu)
sys.modules.setdefault("src.processor.gpu_processor", stub_gpu)


def write_lines(path: Path, lines: list[str]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        fout.write("\n".join(lines))


def test_stream_jsonl_ignores_blank_lines(tmp_path: Path) -> None:
    file_path = tmp_path / "data.jsonl"
    write_lines(
        file_path,
        [
            "",
            json.dumps({"a": 1}),
            "  ",
            json.dumps({"b": 2}),
        ],
    )

    items = list(stream_jsonl_(file_path))
    assert items == [{"a": 1}, {"b": 2}]


def test_stream_jsonl_raises_on_invalid_json(tmp_path: Path) -> None:
    file_path = tmp_path / "broken.jsonl"
    write_lines(file_path, ["not json"])

    with pytest.raises(ValueError) as exc:
        list(stream_jsonl_(file_path))
    assert "JSON parse error" in str(exc.value)


def test_stream_jsonl_requires_object(tmp_path: Path) -> None:
    file_path = tmp_path / "array.jsonl"
    write_lines(file_path, [json.dumps([1, 2, 3])])

    with pytest.raises(TypeError) as exc:
        list(stream_jsonl_(file_path))
    assert "Expected a JSON object" in str(exc.value)
