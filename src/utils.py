import json
from pathlib import Path
from typing import Any, Callable, Iterator, cast
import re
import time


REWRITTEN_CODE_MARKER = "<|REWRITTEN_CODE|>:"


def stream_jsonl(file_path: Path, batch_size: int = 1024) -> Iterator[list[dict[str, Any]]]:
    """Stream JSONL file in batches"""
    batch = []
    with file_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:  # return remaining data
            yield batch


def extract_rewritten_code(text: str, language: str) -> str:
    start_index = text.find(REWRITTEN_CODE_MARKER)
    if start_index == -1:
        pattern = rf"```{language}\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    text = text[start_index + len(REWRITTEN_CODE_MARKER) :].strip()
    pattern = rf"```{language}\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return ""


def split_into_chunks(items: list, n_workers: int) -> list[list]:
    """Split items into roughly equal chunks for workers"""
    chunk_size = len(items) // n_workers
    remainder = len(items) % n_workers

    chunks = []
    start = 0

    for i in range(n_workers):
        # Distribute remainder across first few chunks
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        if current_chunk_size > 0:
            chunks.append(items[start : start + current_chunk_size])
            start += current_chunk_size

    return [chunk for chunk in chunks if chunk]  # Remove empty chunks


def merge_temp_files(temp_files: list[Path], output_path: Path) -> None:
    """Merge all temporary files into the final output file"""
    with output_path.open("w", encoding="utf-8") as fout:
        for temp_file in temp_files:
            if temp_file.exists():
                with temp_file.open("r", encoding="utf-8") as fin:
                    # Copy line by line to handle large files efficiently
                    for line in fin:
                        fout.write(line)
                # Clean up temporary file
                temp_file.unlink()


def process_chunk_to_file(args: tuple[list, Callable[[dict, str, str, Path], dict], Path, int, str, str]):
    """Process a chunk of items and write to a separate file"""
    chunk, process_func, temp_dir, worker_id, input_target_key, output_target_key = args
    temp_file = temp_dir / f"worker_{worker_id}.jsonl"
    start_time = time.time()

    with temp_file.open("w", encoding="utf-8") as fout:
        for item in chunk:
            item = cast(dict, item)
            result = process_func(item, input_target_key, output_target_key, temp_dir)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    processing_time = time.time() - start_time

    return {
        "temp_file": temp_file,
        "items_processed": len(chunk),
        "processing_time": processing_time,
        "worker_id": worker_id,
    }


def have_linter_errors(item: dict) -> bool:
    """Check if the item has linter errors."""
    return len(item.get("lint_report", [])) > 0 if "lint_report" in item else False


def extract_score(text: str) -> int:
    pattern = r"\[\[(\d+)\]\]"
    match = re.search(pattern, text)

    if match:
        return int(match.group(1))
    else:
        return 0


def extract_scores_from_multiple_texts(texts: list[str]) -> list[int]:
    scores = []
    for i, text in enumerate(texts):
        score = extract_score(text)
        scores.append(score)
    return scores
