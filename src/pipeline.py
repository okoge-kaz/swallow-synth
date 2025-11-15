import argparse
import asyncio
from functools import partial
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator

from src.global_vars import get_logger, init_logger
from src.processor.cpu_processor import (
    auto_format,
    filter_by_content_length,
    filter_by_linter_errors,
    split_dataset_by_score,
)
from src.processor.gpu_processor import Processor, llm_rewrite_processor, score_processor
from src.prompts import get_prompt
from src.utils import (
    extract,
    extract_scores_from_multiple_texts,
)


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
    model_name: str,
    tensor_parallel_size: int,
    model_max_length: int,
    prompt_type: str,
    input_target_key: str,
    output_target_key: str,
    backend: str,
    reasoning_effort: str = "high",
    max_num_seqs: int = 20,
) -> None:
    """LLM-based code rewriting using GPU processing"""
    logger = get_logger()
    logger.info(f"llm_rewrite: Model loading: LLM model: {model_name}, backend: {backend}")
    processor = Processor(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=model_max_length,
        backend=backend,
        max_num_seqs=max_num_seqs,
    )
    logger.info(f"llm_rewrite: Model loaded: {model_name}")

    total_items = 0
    start_time = time.perf_counter()

    logger.info(f"Starting LLM rewriting with {tensor_parallel_size} GPUs using {prompt_type} prompt...")
    system_prompt = get_prompt(prompt_type, lang)

    rewrite_proc = partial(
        llm_rewrite_processor,
        input_target_key=input_target_key,
        system_prompt=system_prompt,
        reasoning_effort=reasoning_effort,
    )

    with output_path.open("w", encoding="utf-8") as fout:

        async def _consume() -> None:
            # pipeline.rewrite_codes must be an ASYNC GENERATOR that yields results per item.
            async for ev in processor.process_code(
                stream_jsonl_(input_path), processor=rewrite_proc, max_in_flight=1024
            ):
                if "error" not in ev:
                    item = ev["item"]
                    output_text = ev["result"]
                    item["output"] = output_text
                    item["generator"] = "gpt-oss-120b"

                    try:
                        assistant_output, _ = extract(output_text)
                    except Exception:
                        assistant_output = ""

                    item[output_target_key] = [
                        {
                            "role": "user",
                            "content": assistant_output,
                        },
                    ]

                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fout.flush()  # ensure truly streaming writes

                nonlocal total_items
                total_items += 1

        asyncio.run(_consume())

    actual_time = time.perf_counter() - start_time
    logger.info(f"LLM rewriting completed: {actual_time:.1f}s total ({actual_time / total_items:.3f}s per item)")


def llm_scoring(
    input_path: Path,
    output_path: Path,
    lang: str,
    model_name: str,
    tensor_parallel_size: int,
    model_max_length: int,
    input_target_key: str,
    backend: str,
    max_num_seqs: int,
) -> None:
    """LLM-based code quality scoring using GPU"""
    logger = get_logger()
    processor = Processor(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=model_max_length,
        backend=backend,
        max_num_seqs=max_num_seqs,
    )

    total_items = 0
    start_time = time.perf_counter()
    logger.info(f"Starting LLM scoring with {tensor_parallel_size} GPUs...")

    model_name = os.path.basename(model_name)
    score_key = "score"
    evaluation_key = f"{model_name}_evaluation"

    system_prompt = get_prompt("stage4", lang)

    score_proc = partial(score_processor, input_target_key=input_target_key, system_prompt=system_prompt)

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

    actual_time = time.perf_counter() - start_time
    logger.info(f"LLM scoring completed: {actual_time:.1f}s total ({actual_time / total_items:.3f}s per item)")


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
    subparser.add_argument("--prompt-type", type=str, default="stage4", choices=["stage4", "stage6"])
    subparser.add_argument(
        "--gpu-backend",
        type=str,
        default="vllm",
        choices=["vllm", "tensorrt-llm"],
        help="GPU backend implementation to use",
    )
    subparser.add_argument(
        "--reasoning-effort",
        type=str,
        default="high",
        choices=["low", "medium", "high"],
        help="Level of reasoning effort for LLM rewriting",
    )
    subparser.add_argument(
        "--max-num-seqs",
        type=int,
        help="Maximum number of sequences to process in parallel on GPU",
    )


if __name__ == "__main__":
    logger = init_logger()

    parser = argparse.ArgumentParser(description="Code Quality Pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # CPU subcommand
    cpu_sub = sub.add_parser("cpu", help="CPU-based tasks")
    cpu_parse_args(cpu_sub)

    # GPU subcommand
    gpu_sub = sub.add_parser("gpu", help="GPU-based tasks")
    gpu_parse_args(gpu_sub)

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
    parser.add_argument("--medium-score-threshold", type=int, default=4, help="Medium score threshold")
    parser.add_argument("--high-score-threshold", type=int, default=7, help="High score threshold")

    args = parser.parse_args()

    logger.info(f"Process stage: {args.process_stage} ({args.cmd})")
    match args.process_stage:
        case 1:  # auto-format
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
        case 2:  # filter by length for LLM scoring/rewrite and filtering out with linter errors
            filter_by_content_length(
                input_path=args.input_jsonl,
                output_path=args.output_jsonl,
                language=args.lang,
                input_target_key=args.input_target_key,
                threshold_character_length=args.filter_threshold_length,
                save_longer_samples=True,
            )
        case 3:  # LLM scoring
            llm_scoring(
                input_path=args.input_jsonl,
                output_path=args.output_jsonl,
                lang=args.lang,
                model_name=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                model_max_length=args.model_max_length,
                input_target_key=args.input_target_key,
                backend=args.gpu_backend,
                max_num_seqs=args.max_num_seqs,
            )
            split_dataset_by_score(
                input_path=args.output_jsonl,
                output_path=args.output_jsonl,
                input_target_key=args.output_target_key,  # score
                medium_score_threshold=args.medium_score_threshold,
                high_score_threshold=args.high_score_threshold,
            )
        case 4:  # LLM rewriting
            llm_rewrite(
                input_path=args.input_jsonl,
                output_path=args.output_jsonl,
                lang=args.lang,
                model_name=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                model_max_length=args.model_max_length,
                prompt_type=args.prompt_type,
                input_target_key=args.input_target_key,
                output_target_key=args.output_target_key,
                backend=args.gpu_backend,
                reasoning_effort=args.reasoning_effort,
                max_num_seqs=args.max_num_seqs,
            )
        case 5:  # auto-format after LLM rewriting & filtering out with linter errors
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
            filter_by_linter_errors(
                input_path=args.output_jsonl,
                output_path=args.output_jsonl,
                input_target_key=args.output_target_key,
            )
        case _:
            raise ValueError(f"Unsupported process stage: {args.process_stage}")
    logger.info("Processing completed.")
