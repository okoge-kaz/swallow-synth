import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Any, Iterator
from functools import partial

from processor.cpu_processor import auto_format, process_file_filter, filter_by_content_length
from src.utils import (
    extract_scores_from_multiple_texts,
    extract_rewritten_code,
)
from src.global_vars import init_logger

from src.prompts import get_prompt
from processor.gpu_processor import CodeProcessor, score_processor_stage4, llm_rewrite_processor


def stream_jsonl_(file_path: Path) -> Iterator[dict[str, Any]]:
    """Yield one JSON object (dict) per non-blank line from a UTF-8 .jsonl file."""
    with file_path.open("r", encoding="utf-8") as fin:
        for i, raw in enumerate(fin, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse error at line {i}: {e.msg}") from e
            if not isinstance(obj, dict):
                raise TypeError(f"Expected a JSON object at line {i}, got {type(obj).__name__}")
            yield obj


def llm_rewrite(
    input_path: Path,
    output_path: Path,
    lang: str,
    model_name: str = "qwen-3",
    tensor_parallel_size: int = 1,
    model_max_length: int = 40960,
    prompt_type: str = "stage5",
    code_key: str = "text_formatted",
) -> None:
    """LLM-based code rewriting using GPU processing"""
    processor = CodeProcessor(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=model_max_length,
    )

    total_items = 0
    start_time = time.time()

    print(f"Starting LLM rewriting with {tensor_parallel_size} GPUs using {prompt_type} prompt...")

    system_prompt = get_prompt(prompt_type, lang)

    rewrite_proc = partial(llm_rewrite_processor, value_key=code_key, system_prompt=system_prompt)

    with output_path.open("w", encoding="utf-8") as fout:

        async def _consume() -> None:
            # pipeline.rewrite_codes must be an ASYNC GENERATOR that yields results per item.
            async for ev in processor.process_code(
                stream_jsonl_(input_path), processor=rewrite_proc, max_in_flight=1024
            ):
                if "error" not in ev:
                    item = ev["item"]
                    improved_text = ev["result"]
                    improved_code = extract_rewritten_code(improved_text, language=lang)
                    item["improved_text"] = improved_text
                    item["improved_code"] = improved_code

                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fout.flush()  # ensure truly streaming writes

                nonlocal total_items
                total_items += 1

        asyncio.run(_consume())

    actual_time = time.time() - start_time
    print(f"LLM rewriting completed: {actual_time:.1f}s total ({actual_time / total_items:.3f}s per item)")


def llm_scoring(
    input_path: Path,
    output_path: Path,
    lang: str,
    model_name: str,
    tensor_parallel_size: int,
    compare_model: bool = False,
    model_max_length: int = 40960,
    code_key: str = "text_formatted",
) -> None:
    """LLM-based code quality scoring using GPU processing"""
    processor = CodeProcessor(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=model_max_length,
    )

    total_items = 0
    start_time = time.time()

    print(f"Starting LLM scoring with {tensor_parallel_size} GPUs...")

    model_name = os.path.basename(model_name)
    if compare_model:
        score_key = f"{model_name}_score"
        evaluation_key = f"{model_name}_evaluation"
    else:
        score_key = "score"
        evaluation_key = f"{model_name}_evaluation"

    system_prompt = get_prompt("stage4", lang)

    score_proc = partial(score_processor_stage4, value_key=code_key, system_prompt=system_prompt)

    with output_path.open("w", encoding="utf-8") as fout:

        async def _consume() -> None:
            async for ev in processor.process_code(stream_jsonl_(input_path), processor=score_proc, max_in_flight=1024):
                if "error" not in ev:
                    item = ev["item"]
                    evaluation = ev["result"]
                    score = extract_scores_from_multiple_texts([evaluation])[0]
                    item[score_key] = score
                    item[evaluation_key] = evaluation
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fout.flush()

                nonlocal total_items
                total_items += 1

        asyncio.run(_consume())

    actual_time = time.time() - start_time
    print(f"LLM scoring completed: {actual_time:.1f}s total ({actual_time / total_items:.3f}s per item)")


def after_rewrite_filter(
    input_dir: Path,
    output_dir: Path,
    workers: int | None = None,
) -> None:
    """
    Process all .jsonl files in input_dir and filter out data containing errors.
    Each file is processed independently with multiprocessing support.
    Items are filtered out if they have:
    1. Linter errors, OR
    2. text_formatted length < 10
    Only clean items are saved to output-dir.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .jsonl files in input directory
    jsonl_files = list(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {input_dir}")
        return

    print(f"Found {len(jsonl_files)} .jsonl files to process")

    # Set default workers to CPU count if not specified
    if workers is None:
        workers = cpu_count()

    # Limit workers to not exceed the number of files
    workers = min(workers, len(jsonl_files))

    print(f"Using {workers} workers for parallel processing")

    # Prepare arguments for multiprocessing
    args_list = [(file_path, output_dir) for file_path in jsonl_files]

    # Process files in parallel
    with Pool(workers) as pool:
        results = pool.map(process_file_filter, args_list)

    # Collect statistics from all filtered results
    total_stats = {
        "total_items": 0,
        "linter_errors_count": 0,
        "text_formatted_length_less_than_10": 0,
        "total_filtered": 0,
        "total_errors": 0,
    }

    for result in results:
        print(f"  {result['file_name']}: {result['filtered_items']} clean items, {result['error_items']} filtered out")

        # Accumulate statistics
        for key in total_stats:
            if key in result["stats"]:
                total_stats[key] += result["stats"][key]

        total_stats["total_filtered"] += result["filtered_items"]
        total_stats["total_errors"] += result["error_items"]

    # Print final statistics
    print("After-rewrite filtering completed:")
    print(f"  Total items processed: {total_stats['total_items']}")
    print(f"  Clean items (saved): {total_stats['total_filtered']}")
    print(f"  Items filtered out: {total_stats['total_errors']}")
    print(
        f"  Items with linter errors: {total_stats['linter_errors_count']} ({total_stats['linter_errors_count'] / total_stats['total_items'] * 100:.1f}%)"
    )
    print(
        f"  Items with text_formatted length < 10: {total_stats['text_formatted_length_less_than_10']} ({total_stats['text_formatted_length_less_than_10'] / total_stats['total_items'] * 100:.1f}%)"
    )
    print(f"  Individual filtered files saved to: {output_dir}")


def cpu_parse_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--num-cpu-workers", type=int, default=16, help="Number of CPU workers")
    subparser.add_argument("--read-batch-size", type=int, default=1024, help="Batch size for reading input JSONL")
    subparser.add_argument(
        "--filter-threshold-length", type=int, default=20480, help="Threshold length for filtering out long samples"
    )
    subparser.add_argument(
        "--tmp-dir", type=Path, default=Path("/tmp"), help="Temporary directory for intermediate files"
    )


def gpu_parse_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--model", type=str, required=True, help="hf model identifier or local model path for TensorRT-LLM or vLLM"
    )
    subparser.add_argument("--tensor-parallel-size", type=int, default=1, help="tensor parallel size for GPU tasks")
    subparser.add_argument("--model-max-length", type=int, default=40960, help="Maximum model length")
    subparser.add_argument("--prompt-type", type=str, default="stage5", choices=["stage5", "stage8"])


if __name__ == "__main__":
    logger = init_logger()

    parser = argparse.ArgumentParser(description="Code Quality Pipeline")

    parser.add_argument("--input-jsonl", type=Path, help="Input JSONL file path", required=True)
    parser.add_argument("--output-jsonl", type=Path, help="Output JSONL file path", required=True)
    parser.add_argument("--lang", type=str, help="Programming language (e.g., python, rust, java)", required=True)
    parser.add_argument(
        "--input-target-key", type=str, default="text", help="Key in JSON object to format (default: text)"
    )
    parser.add_argument(
        "--output-target-key",
        type=str,
        default="text_formatted",
        help="Key to store formatted code (default: text_formatted)",
    )
    parser.add_argument("--process-stage", type=int, choices=range(1, 5), required=True)

    sub = parser.add_subparsers(dest="cmd", required=True)

    # CPU subcommand
    cpu_sub = sub.add_parser("cpu", help="CPU-based tasks")
    cpu_parse_args(cpu_sub)

    # GPU subcommand
    gpu_sub = sub.add_parser("gpu", help="GPU-based tasks")
    gpu_parse_args(gpu_sub)

    args = parser.parse_args()

    logger.info(f"Process stage: {args.process_stage} ({args.cmd})")
    match args.process_stage:
        case 1:
            auto_format(
                input_path=args.input_jsonl,
                output_path=args.output_jsonl,
                language=args.lang,
                input_target_key=args.input_target_key,
                output_target_key=args.output_target_key,
                n_workers=args.num_cpu_workers,
                batch_size=args.read_batch_size,
                tmp_dir=args.tmp_dir,
            )
        case 2:
            filter_by_content_length(
                input_path=args.input_jsonl,
                output_path=args.output_jsonl,
                language=args.lang,
                input_target_key=args.input_target_key,
                threshold_character_length=args.filter_threshold_length,
                save_longer_samples=True,
            )
        case 3:
            llm_scoring(
                input_path=args.input_jsonl,
                output_path=args.output_jsonl,
                lang=args.lang,
                model_name=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                model_max_length=args.model_max_length,
                code_key=args.input_target_key,
            )
        case 4:
            llm_rewrite(
                input_path=args.input_jsonl,
                output_path=args.output_jsonl,
                lang=args.lang,
                model_name=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                model_max_length=args.model_max_length,
                prompt_type=args.prompt_type,
                code_key=args.input_target_key,
            )
        case 5:
            pass
        case _:
            raise ValueError(f"Unsupported process stage: {args.process_stage}")
    logger.info("Processing completed.")
