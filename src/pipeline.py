import json
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Callable, Any, Iterator
import tempfile
import os

from src.languages.python import (
    process_item_cpu as python_process_item_cpu,
    RewritePipeline as PythonRewritePipeline,
)


def get_process_item_cpu(lang: str) -> Callable:
    processors = {
        "python": python_process_item_cpu,
    }
    if lang not in processors:
        raise ValueError(f"Unsupported language: {lang}")
    return processors[lang]


def get_rewrite_pipeline(lang: str, model_name: str, batch_size: int, tensor_parallel_size: int = 1):
    pipelines = {
        "python": PythonRewritePipeline,
    }
    if lang not in pipelines:
        raise ValueError(f"Unsupported language: {lang}")
    return pipelines[lang](model_name, tensor_parallel_size)


def process_chunk_to_file(args):
    """Process a chunk of items and write to a separate file"""
    chunk, process_func, temp_dir, worker_id = args

    # Create a unique temporary file for this worker
    temp_file = temp_dir / f"worker_{worker_id}.jsonl"

    start_time = time.time()

    with temp_file.open("w", encoding="utf-8") as fout:
        for item in chunk:
            result = process_func(item)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    processing_time = time.time() - start_time

    return {
        "temp_file": temp_file,
        "items_processed": len(chunk),
        "processing_time": processing_time,
        "worker_id": worker_id,
    }


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


def preprocess(input_path: Path, output_path: Path, lang: str, n_workers: int = 16, batch_size: int = 1000) -> None:
    """CPU only processing stage with separate files per worker"""
    n_workers = n_workers or cpu_count()
    process_item_cpu = get_process_item_cpu(lang)

    # Create temporary directory for worker files
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        total_items = 0
        start_time = time.time()
        temp_files = []

        print(f"Starting CPU processing with {n_workers} workers...")

        with Pool(n_workers) as pool:
            # Process all batches
            for batch in stream_jsonl(input_path, batch_size):
                total_items += len(batch)
                print(f"Processing batch of {len(batch)} items...")

                # Split batch into chunks for workers
                chunks = split_into_chunks(batch, n_workers)

                # Prepare arguments for each worker
                args_list = [(chunk, process_item_cpu, temp_dir, i) for i, chunk in enumerate(chunks)]

                # Process chunks in parallel
                worker_results = pool.map(process_chunk_to_file, args_list)

                # Collect temporary files from this batch
                batch_temp_files = [result["temp_file"] for result in worker_results]

                # Report processing stats
                total_time = sum(result["processing_time"] for result in worker_results)
                avg_time = total_time / len(worker_results) if worker_results else 0
                print(f"  Batch processed in {avg_time:.2f}s average per worker", flush=True)

                # Merge batch results immediately to avoid accumulating too many temp files
                batch_output = temp_dir / f"batch_{len(temp_files)}.jsonl"
                merge_temp_files(batch_temp_files, batch_output)
                temp_files.append(batch_output)

        # Final merge of all batch files
        print(f"Merging {len(temp_files)} batch files...")
        merge_temp_files(temp_files, output_path)

        actual_time = time.time() - start_time
        print(f"CPU processing completed: {actual_time:.1f}s total ({actual_time / total_items:.3f}s per item)")


def rewrite(
    input_path: Path,
    output_path: Path,
    lang: str,
    model_name: str = "qwen-3",
    batch_size: int = 32,
    tensor_parallel_size: int = 1,
) -> None:
    """GPU only processing stage"""
    pipeline = get_rewrite_pipeline(lang, model_name, batch_size, tensor_parallel_size)

    total_items = 0
    start_time = time.time()

    # Process in chunks to manage memory
    chunk_size = 1000  # Process 1000 items at a time
    current_chunk = []

    with output_path.open("w", encoding="utf-8", buffering=1024 * 1024) as fout:  # 1MB buffer
        for batch in stream_jsonl(input_path, batch_size):
            total_items += len(batch)
            results = pipeline.process_batch(batch)

            # Mark as rewritten but not formatted
            for result in results:
                result["status"] = "rewritten"

            current_chunk.extend(results)

            # Write intermediate results when chunk is full
            if len(current_chunk) >= chunk_size:
                for result in current_chunk:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()  # Flush only when chunk is full
                current_chunk = []

        # Write remaining results
        if current_chunk:
            for result in current_chunk:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

    # Process formatting in chunks
    formatted_chunks = []

    with input_path.open("r", encoding="utf-8") as fin:
        current_chunk = []
        for line in fin:
            result = json.loads(line)
            if result["status"] == "rewritten":
                current_chunk.append(result)
                if len(current_chunk) >= chunk_size:
                    # Format chunk
                    formatted_results = []
                    format_errors = []
                    for item in current_chunk:
                        formatted, errors = pipeline.format_code(item["rewritten_text"])
                        formatted_results.append(formatted)
                        format_errors.append(errors)
                    formatted_chunks.append((current_chunk, formatted_results, format_errors))
                    current_chunk = []

        # Process remaining items
        if current_chunk:
            formatted_results = []
            format_errors = []
            for item in current_chunk:
                formatted, errors = pipeline.format_code(item["rewritten_text"])
                formatted_results.append(formatted)
                format_errors.append(errors)
            formatted_chunks.append((current_chunk, formatted_results, format_errors))

    # Process second rewrite if needed
    needs_second_rewrite = []
    second_rewrite_indices = []

    for chunk, formatted_results, format_errors in formatted_chunks:
        for i, errors in enumerate(format_errors):
            if errors:
                needs_second_rewrite.append({"text": formatted_results[i], "lint_report": json.dumps(errors)})
                second_rewrite_indices.append((chunk, i))

    # Perform second rewrite only for items with errors
    if needs_second_rewrite:
        second_rewritten = []
        for i in range(0, len(needs_second_rewrite), batch_size):
            batch = needs_second_rewrite[i : i + batch_size]
            results = pipeline.process_batch(batch)
            second_rewritten.extend(results)

        # Format only the second rewritten items
        for (chunk, idx), rewritten in zip(second_rewrite_indices, second_rewritten):
            final_code, final_errors = pipeline.format_code(rewritten["rewritten_text"])
            chunk[idx]["rewritten_text"] = final_code
            if final_errors:
                chunk[idx]["format_errors"] = final_errors

    # Write final results
    with output_path.open("w", encoding="utf-8", buffering=1024 * 1024) as fout:
        for chunk, formatted_results, _ in formatted_chunks:
            for item, formatted in zip(chunk, formatted_results):
                item["rewritten_text"] = formatted
                item["status"] = "formatted"
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()  # Flush after each chunk

    actual_time = time.time() - start_time
    print(f"GPU processing time: {actual_time:.1f}s (1 item: {actual_time / total_items:.3f}s)")


# === CLI Entrypoint ===

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Code Quality Pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("cpu", help="CPU only processing stage")
    p1.add_argument("--input-jsonl", type=Path, required=True)
    p1.add_argument("--output-jsonl", type=Path, required=True)
    p1.add_argument("--workers", type=int, default=32, help="Number of CPU workers")
    p1.add_argument("--lang", type=str, required=True, help="Programming language (e.g., python, rust, java)")
    p1.add_argument("--batch-size", type=int, default=1000, help="Batch size for CPU processing")

    p2 = sub.add_parser("gpu", help="GPU only processing stage")
    p2.add_argument("--input-jsonl", type=Path, required=True)
    p2.add_argument("--output-jsonl", type=Path, required=True)
    p2.add_argument("--model", type=str, default="qwen-3", help="Local Qwen model identifier for vLLM")
    p2.add_argument("--lang", type=str, required=True, help="Programming language (e.g., python, rust, java)")
    p2.add_argument("--batch-size", type=int, default=32, help="Batch size for GPU processing")
    p2.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use for tensor parallelism")

    args = parser.parse_args()
    if args.cmd == "cpu":
        preprocess(
            input_path=args.input_jsonl,
            output_path=args.output_jsonl,
            n_workers=args.workers,
            lang=args.lang,
            batch_size=args.batch_size,
        )
    elif args.cmd == "gpu":
        rewrite(
            input_path=args.input_jsonl,
            output_path=args.output_jsonl,
            model_name=args.model,
            lang=args.lang,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
        )
