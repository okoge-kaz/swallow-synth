import json
from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
import tempfile
from typing import Callable

from src.global_vars import get_logger
from src.languages.c import (
    process_item_cpu as c_process_item_cpu,
)
from src.languages.cpp import (
    process_item_cpu as cpp_process_item_cpu,
)
from src.languages.cuda import (
    process_item_cpu as cuda_process_item_cpu,
)
from src.languages.go import (
    process_item_cpu as go_process_item_cpu,
)
from src.languages.javascript import (
    process_item_cpu as javascript_process_item_cpu,
)
from src.languages.python import (
    process_item_cpu as python_process_item_cpu,
)
from src.languages.rust import (
    process_item_cpu as rust_process_item_cpu,
)
from src.languages.typescript import (
    process_item_cpu as typescript_process_item_cpu,
)
from src.utils import (
    have_linter_errors,
    merge_temp_files,
    process_chunk_to_file,
    split_into_chunks,
    stream_jsonl,
)


def get_process_item_cpu(language: str) -> Callable:
    processors = {
        "python": python_process_item_cpu,
        "c": c_process_item_cpu,
        "cpp": cpp_process_item_cpu,
        "cuda": cuda_process_item_cpu,
        "go": go_process_item_cpu,
        "rust": rust_process_item_cpu,
        "javascript": javascript_process_item_cpu,
        "typescript": typescript_process_item_cpu,
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

            if have_linter_errors(item=item):
                continue

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


def filter_by_linter_errors(
    input_path: Path,
    output_path: Path,
    input_target_key: str,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist")

    logger = get_logger()
    logger.info(f"Filtering samples in '{input_target_key}' by linter errors")

    valid_samples = []
    error_samples = []
    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            item: dict[str, str] = json.loads(line)
            assert input_target_key in item, f"Key '{input_target_key}' not found in item: {item}"

            if have_linter_errors(item=item):
                error_samples.append(item)
            else:
                valid_samples.append(item)

    valid_samples_path = output_path

    with valid_samples_path.open("w", encoding="utf-8") as fout:
        logger.info(f"Saving {len(valid_samples)} valid samples to {valid_samples_path}")
        for item in valid_samples:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(
        f"Filtering by linter errors completed. Total samples: {len(valid_samples) + len(error_samples)}, samples with linter errors: {len(error_samples)}, valid samples: {len(valid_samples)}"
    )


def split_dataset_by_score(
    input_path: Path,
    output_path: Path,
    input_target_key: str,
    medium_score_threshold: int,
    high_score_threshold: int,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist")

    logger = get_logger()
    logger.info(
        f"Splitting dataset by score thresholds: medium >= {medium_score_threshold}, high >= {high_score_threshold}"
    )

    low_score_samples = []
    medium_score_samples = []
    high_score_samples = []

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            item: dict[str, str] = json.loads(line)
            assert input_target_key in item, f"Key '{input_target_key}' not found in item: {item}"

            score = item.get("score", 0)
            score = int(score)
            if score >= high_score_threshold:
                high_score_samples.append(item)
            elif score >= medium_score_threshold:
                medium_score_samples.append(item)
            else:
                low_score_samples.append(item)

    output_dir = output_path.parent
    os.makedirs(output_dir, exist_ok=True)

    low_score_path = output_dir / "low" / output_path.name
    medium_score_path = output_dir / "medium" / output_path.name
    high_score_path = output_dir / "high" / output_path.name
    os.makedirs(low_score_path.parent, exist_ok=True)
    os.makedirs(medium_score_path.parent, exist_ok=True)
    os.makedirs(high_score_path.parent, exist_ok=True)

    with low_score_path.open("w", encoding="utf-8") as fout:
        logger.info(f"Saving {len(low_score_samples)} low score samples to {low_score_path}")
        for item in low_score_samples:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    with medium_score_path.open("w", encoding="utf-8") as fout:
        logger.info(f"Saving {len(medium_score_samples)} medium score samples to {medium_score_path}")
        for item in medium_score_samples:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    with high_score_path.open("w", encoding="utf-8") as fout:
        logger.info(f"Saving {len(high_score_samples)} high score samples to {high_score_path}")
        for item in high_score_samples:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(
        f"Dataset splitting completed. Total samples: {len(low_score_samples) + len(medium_score_samples) + len(high_score_samples)}, "
        f"low score samples: {len(low_score_samples)}, medium score samples: {len(medium_score_samples)}, high score samples: {len(high_score_samples)}"
    )
