from typing import Callable
import os
from pathlib import Path
import json
import tempfile

from multiprocessing import Pool, cpu_count

from src.utils import (
    stream_jsonl,
    split_into_chunks,
    process_chunk_to_file,
    merge_temp_files,
    have_linter_errors,
)

from src.languages.python import (
    process_item_cpu as python_process_item_cpu,
)
from src.global_vars import get_logger


def get_process_item_cpu(language: str) -> Callable:
    processors = {
        "python": python_process_item_cpu,
    }
    if language not in processors:
        raise ValueError(f"Unsupported language: {language}")
    return processors[language]


def auto_format(
    input_path: Path,
    output_path: Path,
    language: str,
    input_target_key: str,
    output_target_key: str,
    n_workers: int,
    batch_size: int,
    tmp_dir: Path,
) -> None:
    n_workers = n_workers or cpu_count()
    process_item_cpu = get_process_item_cpu(language)
    logger = get_logger()

    with tempfile.TemporaryDirectory(dir=tmp_dir) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        temp_files = []

        logger.info(
            f"Formatting language: {language}: processing '{input_target_key}' -> '{output_target_key}', using {n_workers} workers"
        )

        with Pool(n_workers) as pool:
            for batch in stream_jsonl(input_path, batch_size):
                chunks = split_into_chunks(batch, n_workers)
                args_list = [
                    (chunk, process_item_cpu, temp_dir, i, input_target_key, output_target_key)
                    for i, chunk in enumerate(chunks)
                ]

                worker_results = pool.map(func=process_chunk_to_file, iterable=args_list)

                batch_temp_files = [result["temp_file"] for result in worker_results]
                total_time = sum(result["processing_time"] for result in worker_results)
                avg_time = total_time / len(worker_results) if worker_results else 0
                logger.info(f"  Batch processed in {avg_time:.2f}s average per worker")
                batch_output = temp_dir / f"batch_{len(temp_files)}.jsonl"
                merge_temp_files(batch_temp_files, batch_output)
                temp_files.append(batch_output)

        logger.info(f"Merging {len(temp_files)} batch files into final output {output_path}")
        merge_temp_files(temp_files, output_path)


def filter_by_content_length(
    input_path: Path,
    output_path: Path,
    language: str,
    input_target_key: str,
    threshold_character_length: int,
    save_longer_samples: bool,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist")

    logger = get_logger()
    logger.info(f"Filtering samples in '{input_target_key}' by length <= {threshold_character_length} characters")

    longer_samples = []
    samples = []
    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            item: dict[str, str] = json.loads(line)
            assert input_target_key in item, f"Key '{input_target_key}' not found in item: {item}"
            text: str = item.get(input_target_key, "")

            if len(text) > threshold_character_length:
                longer_samples.append(item)
            else:
                samples.append(item)

    longer_samples_path = output_path.parent / "longer_samples" / output_path.name
    samples_path = output_path
    os.makedirs(longer_samples_path.parent, exist_ok=True)

    if save_longer_samples:
        logger.info(f"Saving {len(longer_samples)} longer samples to {longer_samples_path}")
        with longer_samples_path.open("w", encoding="utf-8") as fout:
            for item in longer_samples:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    with samples_path.open("w", encoding="utf-8") as fout:
        logger.info(f"Saving {len(samples)} regular samples to {samples_path}")
        for item in samples:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(
        f"Filtering by content length completed. Total samples: {len(samples) + len(longer_samples)}, longer samples: {len(longer_samples)}, samples: {len(samples)}"
    )


def process_file_filter(args):
    """Process a single file and filter out error-containing data"""
    file_path, output_dir = args

    filtered_items = []
    file_stats = {"total_items": 0, "linter_errors_count": 0, "text_formatted_length_less_than_10": 0}

    print(f"Processing {file_path.name}...")

    with file_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            item = json.loads(line)
            file_stats["total_items"] += 1

            # Check for various error conditions
            text_formatted = item.get("text_formatted", "")

            # Check conditions
            has_linter_errors = have_linter_errors(item)
            text_formatted_long_enough = len(text_formatted) >= 10

            # Count statistics
            if has_linter_errors:
                file_stats["linter_errors_count"] += 1
            if not text_formatted_long_enough:
                file_stats["text_formatted_length_less_than_10"] += 1

            # Only keep items that pass all quality checks
            if not has_linter_errors and text_formatted_long_enough:
                filtered_items.append(item)

    # Write filtered results for this file with train_ prefix
    file_stem = file_path.stem
    filtered_output_path = output_dir / f"{file_stem}.jsonl"

    with filtered_output_path.open("w", encoding="utf-8") as fout:
        for item in filtered_items:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "file_name": file_path.name,
        "filtered_items": len(filtered_items),
        "error_items": file_stats["total_items"] - len(filtered_items),
        "filtered_output_path": filtered_output_path,
        "stats": file_stats,
    }
