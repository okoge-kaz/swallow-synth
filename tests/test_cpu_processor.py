import json
from pathlib import Path
from typing import Any, Dict

import pytest

from src.processor import cpu_processor


class DummyLogger:
    def info(self, *_args: Any, **_kwargs: Any) -> None:
        pass


@pytest.fixture(autouse=True)
def mock_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cpu_processor, "get_logger", lambda: DummyLogger())


def write_jsonl(path: Path, rows: list[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fin:
        return [json.loads(line) for line in fin]


@pytest.mark.parametrize(
    "language,expected",
    [
        ("python", cpu_processor.python_process_item_cpu),
        ("c", cpu_processor.c_process_item_cpu),
        ("cpp", cpu_processor.cpp_process_item_cpu),
        ("cuda", cpu_processor.cuda_process_item_cpu),
        ("go", cpu_processor.go_process_item_cpu),
        ("rust", cpu_processor.rust_process_item_cpu),
        ("javascript", cpu_processor.javascript_process_item_cpu),
        ("typescript", cpu_processor.typescript_process_item_cpu),
    ],
)
def test_get_process_item_cpu_known_language(language, expected) -> None:
    processor = cpu_processor.get_process_item_cpu(language)
    assert processor is expected


def test_get_process_item_cpu_unknown_language() -> None:
    with pytest.raises(ValueError):
        cpu_processor.get_process_item_cpu("ruby")


def test_filter_by_content_length(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    rows = [
        {"code": "short", "lint_report": []},
        {"code": "x" * 50, "lint_report": []},
        {"code": "ignored", "lint_report": [{"rule": "E999"}]},
    ]
    write_jsonl(input_path, rows)

    cpu_processor.filter_by_content_length(
        input_path=input_path,
        output_path=output_path,
        language="python",
        input_target_key="code",
        threshold_character_length=10,
        save_longer_samples=True,
    )

    regular_rows = read_jsonl(output_path)
    assert len(regular_rows) == 1
    assert regular_rows[0]["code"] == "short"

    longer_path = output_path.parent / "longer_samples" / output_path.name
    longer_rows = read_jsonl(longer_path)
    assert len(longer_rows) == 1
    assert longer_rows[0]["code"] == "x" * 50


def test_filter_by_linter_errors(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "clean.jsonl"

    rows = [
        {"code": "valid", "lint_report": []},
        {"code": "bad", "lint_report": [{"rule": "E123"}]},
    ]
    write_jsonl(input_path, rows)

    cpu_processor.filter_by_linter_errors(
        input_path=input_path,
        output_path=output_path,
        input_target_key="code",
    )

    result_rows = read_jsonl(output_path)
    assert result_rows == [{"code": "valid", "lint_report": []}]


def test_split_dataset_by_score(tmp_path: Path) -> None:
    input_path = tmp_path / "scores.jsonl"
    output_path = tmp_path / "split.jsonl"

    rows = [
        {"score": 2},
        {"score": 5},
        {"score": 9},
    ]
    write_jsonl(input_path, rows)

    cpu_processor.split_dataset_by_score(
        input_path=input_path,
        output_path=output_path,
        input_target_key="score",
        medium_score_threshold=4,
        high_score_threshold=7,
    )

    low_rows = read_jsonl(output_path.parent / "low" / output_path.name)
    medium_rows = read_jsonl(output_path.parent / "medium" / output_path.name)
    high_rows = read_jsonl(output_path.parent / "high" / output_path.name)

    assert low_rows == [{"score": 2}]
    assert medium_rows == [{"score": 5}]
    assert high_rows == [{"score": 9}]
