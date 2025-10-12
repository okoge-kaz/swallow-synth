import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import pytest

from src.utils import (
    REWRITTEN_CODE_MARKER,
    apply_chat_template,
    extract_rewritten_code,
    extract_score,
    extract_scores_from_multiple_texts,
    have_linter_errors,
    merge_temp_files,
    process_chunk_to_file,
    split_into_chunks,
    stream_jsonl,
)


# -----------------------------
# stream_jsonl
# -----------------------------
def test_stream_jsonl_basic(tmp_path: Path) -> None:
    p = tmp_path / "input.jsonl"
    rows = [{"i": i} for i in range(10)]
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    batches = list(stream_jsonl(p, batch_size=3))
    # 3,3,3,1 (4 batches)
    assert [len(b) for b in batches] == [3, 3, 3, 1]
    # check content
    assert batches[0][0]["i"] == 0
    assert batches[-1][-1]["i"] == 9


def test_stream_jsonl_handles_final_partial_batch(tmp_path: Path) -> None:
    p = tmp_path / "input.jsonl"
    rows = [{"x": "last batch test"} for _ in range(1)]
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    batches = list(stream_jsonl(p, batch_size=1024))
    assert len(batches) == 1
    assert batches[0] == rows


# -----------------------------
# extract_rewritten_code
# -----------------------------
@pytest.mark.parametrize(
    "text,language,expected",
    [
        # No marker, single code block
        ("before\n```python\nprint('ok')\n```\nafter", "python", "print('ok')"),
        # Marker, single code block
        (
            f"xxxx\n{REWRITTEN_CODE_MARKER}\nComment\n```python\nx=1\n```\n```python\nx=2\n```",
            "python",
            "x=1",
        ),
        # other language code block
        ("```javascript\nlet x = 1;\n```\n", "python", ""),
        (f"{REWRITTEN_CODE_MARKER}\nno code here", "python", ""),
        ("plain text only", "python", ""),
        ("```python\n  a = 1  \n\n```\n", "python", "a = 1"),
    ],
)
def test_extract_rewritten_code(text: str, language: str, expected: str) -> None:
    assert extract_rewritten_code(text, language) == expected


def test_extract_rewritten_code_prefers_after_marker() -> None:
    # code block exists before the marker, but we prefer the one after
    text = f"```python\nbefore=1\n```\n{REWRITTEN_CODE_MARKER}\n```python\nafter=2\n```\n"
    assert extract_rewritten_code(text, "python") == "after=2"


# -----------------------------
# split_into_chunks
# -----------------------------
@pytest.mark.parametrize(
    "items,n_workers,shapes",
    [
        (list(range(10)), 3, [4, 3, 3]),  # normal case
        (list(range(5)), 10, [1, 1, 1, 1, 1]),  # the number of workers exceeds items
        ([], 4, []),  # empty list
        (list(range(9)), 2, [5, 4]),
    ],
)
def test_split_into_chunks(items: List[int], n_workers: int, shapes: List[int]) -> None:
    chunks = split_into_chunks(items, n_workers)
    assert [len(c) for c in chunks] == shapes
    # check all items are included
    assert sum(len(c) for c in chunks) == len(items)
    # check order is preserved
    flat = [x for c in chunks for x in c]
    assert flat == items


# -----------------------------
# merge_temp_files
# -----------------------------
def test_merge_temp_files_merges_and_deletes(tmp_path: Path) -> None:
    temp1 = tmp_path / "t1.jsonl"
    temp2 = tmp_path / "t2.jsonl"
    for p, rows in [(temp1, [{"a": 1}, {"a": 2}]), (temp2, [{"b": 3}])]:
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out = tmp_path / "out.jsonl"
    merge_temp_files([temp1, temp2], out)

    # check output content
    with out.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert lines == [{"a": 1}, {"a": 2}, {"b": 3}]

    # check temp files are deleted
    assert not temp1.exists()
    assert not temp2.exists()


# -----------------------------
# process_chunk_to_file
# -----------------------------
def test_process_chunk_to_file_writes_and_reports(tmp_path: Path) -> None:
    # dumpy processing function
    def dummy_process_func(item: Dict[str, Any], in_key: str, out_key: str, _tmp: Path) -> Dict[str, Any]:
        text = cast(str, item.get(in_key, ""))
        return {**item, out_key: text.upper()}

    chunk = [
        {"id": 1, "input": "hello"},
        {"id": 2, "input": "world"},
    ]
    args: Tuple[List[Dict[str, Any]], Any, Path, int, str, str] = (
        chunk,
        dummy_process_func,
        tmp_path,
        0,
        "input",
        "output",
    )
    report = process_chunk_to_file(args)

    assert report["worker_id"] == 0
    assert report["items_processed"] == 2
    assert report["processing_time"] >= 0.0
    temp_file: Path = report["temp_file"]
    assert temp_file.exists()

    with temp_file.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert rows == [
        {"id": 1, "input": "hello", "output": "HELLO"},
        {"id": 2, "input": "world", "output": "WORLD"},
    ]


# -----------------------------
# have_linter_errors
# -----------------------------
@pytest.mark.parametrize(
    "item,expected",
    [
        ({}, False),
        ({"lint_report": []}, False),
        ({"lint_report": [{"rule": "E999"}]}, True),
    ],
)
def test_have_linter_errors(item: Dict[str, Any], expected: bool) -> None:
    assert have_linter_errors(item) == expected


# -----------------------------
# extract_score / extract_scores_from_multiple_texts
# -----------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Score: [[12]] points", 12),
        ("No score here", 0),
        ("[[001]] leading zeros", 1),
        ("prefix [[99]] suffix [[100]]", 99),
    ],
)
def test_extract_score(text: str, expected: int) -> None:
    assert extract_score(text) == expected


def test_extract_scores_from_multiple_texts() -> None:
    texts = ["[[3]]", "no", "score [[7]]", "[[0]]"]
    assert extract_scores_from_multiple_texts(texts) == [3, 0, 7, 0]


# -----------------------------
# apply_chat_template
# -----------------------------
class DummyTokenizer:
    """transformers.PreTrainedTokenizer dummy implementation."""

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        sys = next(m["content"] for m in conversation if m["role"] == "system")
        usr = next(m["content"] for m in conversation if m["role"] == "user")
        return f"<s>[SYS]{sys}[/SYS]\n[USR]{usr}\n[GEN]"


@pytest.mark.parametrize(
    "system_prompt,user_input",
    [
        ("You are a coder.", "Rewrite this code."),
        ("日本語で回答してください。", "次のコードを改善してください。"),
    ],
)
def test_apply_chat_template(system_prompt: str, user_input: str) -> None:
    tok = cast(Any, DummyTokenizer())
    out = apply_chat_template(tok, system_prompt, user_input)
    assert isinstance(out, str)
    assert "[SYS]" in out and "[USR]" in out and "[GEN]" in out
    assert system_prompt in out and user_input in out
