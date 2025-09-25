import asyncio
import json
import re
import os
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Callable, Any, Iterator
import tempfile

from src.languages.python import (
    process_item_cpu as python_process_item_cpu,
    PythonRewritePipeline,
)
from src.languages.abc import RewritePipeline
from src.languages.finemath import math_rewrite


def get_process_item_cpu(lang: str) -> Callable:
    processors = {
        "python": python_process_item_cpu,
    }
    if lang not in processors:
        raise ValueError(f"Unsupported language: {lang}")
    return processors[lang]


def get_rewrite_pipeline(
    lang: str, model_name: str, tensor_parallel_size: int = 1, model_max_length: int = 131072, use_async=False
) -> RewritePipeline:
    pipelines = {
        "python": PythonRewritePipeline,
    }
    if lang not in pipelines:
        raise ValueError(f"Unsupported language: {lang}")
    return pipelines[lang](model_name, tensor_parallel_size, model_max_length, use_async)


def process_chunk_to_file(args):
    """Process a chunk of items and write to a separate file"""
    chunk, process_func, temp_dir, worker_id, target_key = args

    # Create a unique temporary file for this worker
    temp_file = temp_dir / f"worker_{worker_id}.jsonl"

    start_time = time.time()

    with temp_file.open("w", encoding="utf-8") as fout:
        for item in chunk:
            result = process_func(item, target_key)
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


def auto_format(
    input_path: Path,
    output_path: Path,
    lang: str,
    target_key: str = "text",
    n_workers: int = 16,
    batch_size: int = 1000,
) -> None:
    """Auto-format code in the specified key using CPU processing"""
    n_workers = n_workers or cpu_count()
    process_item_cpu = get_process_item_cpu(lang)

    # Create temporary directory for worker files
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        total_items = 0
        start_time = time.time()
        temp_files = []

        print(f"Starting auto-format processing with {n_workers} workers on key '{target_key}'...")

        with Pool(n_workers) as pool:
            # Process all batches
            for batch in stream_jsonl(input_path, batch_size):
                total_items += len(batch)
                print(f"Processing batch of {len(batch)} items...")

                # Split batch into chunks for workers
                chunks = split_into_chunks(batch, n_workers)

                # Prepare arguments for each worker
                args_list = [(chunk, process_item_cpu, temp_dir, i, target_key) for i, chunk in enumerate(chunks)]

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
        print(f"Auto-format processing completed: {actual_time:.1f}s total ({actual_time / total_items:.3f}s per item)")


def extract_rewritten_code(text: str) -> str:
    import re

    start_marker = "<|REWRITTEN_CODE|>:"
    start_index = text.find(start_marker)
    if start_index == -1:
        pattern = r"```python\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    text = text[start_index + len(start_marker) :]

    pattern = r"```python\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_generated_code(text: str) -> str:
    import re

    start_marker = "<|GENERATED_CODE|>:"
    start_index = text.find(start_marker)
    if start_index == -1:
        return ""

    text = text[start_index + len(start_marker) :]

    pattern = r"```python\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def llm_rewrite(
    input_path: Path,
    output_path: Path,
    lang: str,
    model_name: str = "qwen-3",
    batch_size: int = 32,
    tensor_parallel_size: int = 1,
    model_max_length: int = 40960,
    prompt_type: str = "stage5",
) -> None:
    """LLM-based code rewriting using GPU processing"""
    pipeline = get_rewrite_pipeline(
        lang=lang,
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        model_max_length=model_max_length,
        use_async=True,
    )

    total_items = 0
    start_time = time.time()

    print(f"Starting LLM rewriting with {tensor_parallel_size} GPUs using {prompt_type} prompt...")

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:

        async def _consume() -> None:
            # pipeline.rewrite_codes must be an ASYNC GENERATOR that yields results per item.
            async for ev in pipeline.rewrite_codes(stream_jsonl_(input_path), prompt_type=prompt_type):
                if "error" not in ev:
                    item = ev["item"]
                    improved_text = ev["result"]
                    improved_code = extract_rewritten_code(improved_text)
                    item["improved_text"] = improved_text
                    item["improved_code"] = improved_code

                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fout.flush()  # ensure truly streaming writes

                nonlocal total_items
                total_items += 1

        asyncio.run(_consume())

    actual_time = time.time() - start_time
    print(f"LLM rewriting completed: {actual_time:.1f}s total ({actual_time / total_items:.3f}s per item)")


def competitive_programming_write(
    input_path: Path,
    output_path: Path,
    lang: str,
    model_name: str = "qwen-3",
    batch_size: int = 32,
    tensor_parallel_size: int = 1,
    model_max_length: int = 40960,
) -> None:
    """LLM-based code rewriting for competitive programming using GPU processing"""
    pipeline = get_rewrite_pipeline(
        lang=lang, model_name=model_name, tensor_parallel_size=tensor_parallel_size, model_max_length=model_max_length
    )

    total_items = 0
    start_time = time.time()

    print(f"Starting competitive programming LLM rewriting with {tensor_parallel_size} GPUs...")

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for batch in stream_jsonl(input_path, batch_size):
            total_items += len(batch)
            print(f"Processing batch of {len(batch)} items...")

            # key in batch should be "text_formatted" for rewriting
            if not all("question" in item for item in batch):
                raise ValueError("All items in the batch must contain 'question' key for code generation")
            questions = [item.get("question", "") for item in batch]

            # Call pipeline.rewrite_codes
            try:
                generated_texts = pipeline.competitive_programming_write(questions)

                # Write results to output file
                for index, item in enumerate(batch):
                    generated_text = generated_texts[index] if index < len(generated_texts) else ""
                    generated_code = extract_generated_code(generated_text)
                    item["generated_text"] = generated_text
                    item["text"] = generated_code
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error during rewriting: {e}")

    actual_time = time.time() - start_time
    print(
        f"Competitive programming LLM rewriting completed: {actual_time:.1f}s total ({actual_time / total_items:.3f}s per item)"
    )


def have_linter_errors(item: dict) -> bool:
    """Check if the item has linter errors."""
    return len(item.get("lint_report", [])) > 0 if "lint_report" in item else False


def separate_code_samples(
    input_path: Path,
    output_path: Path,
    tokenizer_path: Path,
    threshold_length: int = 20480,
) -> None:
    """Separate code samples with and without linter errors based on a threshold length."""
    import json

    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist")

    longer_samples = []
    samples = []
    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            item = json.loads(line)
            assert "text_formatted" in item, "Each item must contain 'text_formatted' key for length checking"
            text = item.get("text_formatted")
            if len(text) >= threshold_length * 2:
                longer_samples.append(item)
            else:
                samples.append(item)

    longer_samples_path = output_path.parent / f"{output_path.stem}_longer_samples.jsonl"
    samples_path = output_path

    with longer_samples_path.open("w", encoding="utf-8") as fout:
        for item in longer_samples:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    with samples_path.open("w", encoding="utf-8") as fout:
        for item in samples:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Separated {len(longer_samples)} longer samples and {len(samples)} regular samples")


def llm_auto_fix(
    input_path: Path,
    output_dir: Path,
    lang: str,
    model_name: str = "qwen-3",
    batch_size: int = 32,
    tensor_parallel_size: int = 1,
    model_max_length: int = 40960,
) -> None:
    import re

    pipeline = get_rewrite_pipeline(
        lang=lang, model_name=model_name, tensor_parallel_size=tensor_parallel_size, model_max_length=model_max_length
    )

    # Extract file number from input_path (e.g., train_0004.jsonl -> 0004)
    file_number_match = re.search(r"(\d+)", input_path.name)
    if not file_number_match:
        raise ValueError(f"Cannot extract file number from {input_path.name}")
    file_number = file_number_match.group(1).zfill(4)

    # Create temporary files
    tmp_without_errors_path = output_dir / f"tmp_{file_number}_without_errors.jsonl"
    tmp_with_errors_path = output_dir / f"tmp_{file_number}_with_errors.jsonl"
    stats_path = output_dir / f"{file_number}.out"

    # Load data and separate with/without errors
    with_errors_data = []
    without_errors_data = []

    print(f"Reading input file: {input_path}")
    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            item = json.loads(line)
            if have_linter_errors(item):
                with_errors_data.append(item)
            else:
                without_errors_data.append(item)

    # Write temporary files
    with tmp_without_errors_path.open("w", encoding="utf-8") as fout:
        for item in without_errors_data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    with tmp_with_errors_path.open("w", encoding="utf-8") as fout:
        for item in with_errors_data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write initial statistics
    with stats_path.open("w", encoding="utf-8") as fout:
        fout.write("Initial statistics:\n")
        fout.write(f"Total items: {len(with_errors_data) + len(without_errors_data)}\n")
        fout.write(f"Without errors: {len(without_errors_data)}\n")
        fout.write(f"With errors: {len(with_errors_data)}\n\n")

    print(f"Separated {len(with_errors_data)} items with errors and {len(without_errors_data)} items without errors")

    # Process with_errors data
    if with_errors_data:
        # Filter by character count
        filtered_data = []
        skipped_count = 0

        for item in with_errors_data:
            text = item.get("text", "")
            if len(text) >= model_max_length:
                skipped_count += 1
                continue
            filtered_data.append(item)

        print(f"Filtered data: {len(filtered_data)} items to process, {skipped_count} items skipped due to length")

        # Process in batches
        fixed_data = []
        if filtered_data:
            print("Starting error fixing process...")

            for i in range(0, len(filtered_data), batch_size):
                batch = filtered_data[i : i + batch_size]
                print(f"Processing batch {i // batch_size + 1}/{(len(filtered_data) + batch_size - 1) // batch_size}")

                # Prepare codes and lint_reports for fix_errors
                codes = []
                lint_reports = []

                for item in batch:
                    codes.append(item.get("text", ""))
                    lint_report = item.get("lint_report", [])
                    if isinstance(lint_report, list):
                        lint_reports.append(json.dumps(lint_report))
                    else:
                        lint_reports.append(str(lint_report))

                # Call pipeline.fix_errors
                try:
                    fixed_codes = pipeline.fix_errors(codes, lint_reports)

                    # Extract code from markdown code blocks if present
                    def extract_code_from_markdown(text: str, lang: str) -> str:
                        code_block_marker = f"```{lang}"

                        if code_block_marker in text:
                            # Find the first ```{lang} block
                            start_marker = code_block_marker
                            end_marker = "```"

                            start_idx = text.find(start_marker)
                            if start_idx != -1:
                                # Move past the start marker and any newline
                                code_start = start_idx + len(start_marker)
                                if code_start < len(text) and text[code_start] == "\n":
                                    code_start += 1

                                # Find the closing ```
                                end_idx = text.find(end_marker, code_start)
                                if end_idx != -1:
                                    return text[code_start:end_idx].strip()

                        # If no code block found, return original text
                        return text

                    # Update items with fixed codes
                    for j, fixed_code in enumerate(fixed_codes):
                        extracted_code = extract_code_from_markdown(fixed_code, lang)
                        batch[j]["auto_fix_output"] = extracted_code  # Store extracted code
                        batch[j]["text"] = extracted_code  # Store extracted code

                    fixed_data.extend(batch)

                except Exception as e:
                    print(f"Error processing batch: {e}")
                    # Keep original data if fixing fails
                    fixed_data.extend(batch)

            print("[INFO]: Error fixing completed")

        # Re-run auto_format on fixed data
        if fixed_data:
            print("Re-running auto_format on fixed data...")

            # Create temporary file for fixed data
            temp_fixed_path = output_dir / f"temp_fixed_{file_number}.jsonl"
            with temp_fixed_path.open("w", encoding="utf-8") as fout:
                for item in fixed_data:
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            # Run auto_format
            temp_reformatted_path = output_dir / f"temp_reformatted_{file_number}.jsonl"
            auto_format(temp_fixed_path, temp_reformatted_path, lang)

            # Categorize results
            still_with_errors = []
            newly_fixed = []

            with temp_reformatted_path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    item = json.loads(line)
                    if have_linter_errors(item):
                        still_with_errors.append(item)
                    else:
                        newly_fixed.append(item)

            # Update tmp_with_errors.jsonl with items that still have errors
            with tmp_with_errors_path.open("w", encoding="utf-8") as fout:
                for item in still_with_errors:
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            # Merge newly fixed items with tmp_without_errors.jsonl
            final_without_errors_path = output_dir / f"train_{file_number}_without_errors.jsonl"
            with final_without_errors_path.open("w", encoding="utf-8") as fout:
                # Write original without_errors_data with auto_fix_output key
                for item in without_errors_data:
                    item["auto_fix_output"] = ""  # Add empty auto_fix_output for consistency
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                # Write newly fixed data
                for item in newly_fixed:
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            # Update statistics
            with stats_path.open("a", encoding="utf-8") as fout:
                fout.write("After error fixing:\n")
                fout.write(f"Items processed for fixing: {len(filtered_data)}\n")
                fout.write(f"Items skipped due to length: {skipped_count}\n")
                fout.write(f"Successfully fixed: {len(newly_fixed)}\n")
                fout.write(f"Still with errors: {len(still_with_errors)}\n")
                fout.write(f"Final without errors: {len(without_errors_data) + len(newly_fixed)}\n")

            # Clean up temporary files
            temp_fixed_path.unlink(missing_ok=True)
            temp_reformatted_path.unlink(missing_ok=True)

            print(f"Completed: {len(newly_fixed)} items fixed, {len(still_with_errors)} items still have errors")
            print(f"Final results saved to: {final_without_errors_path}")
            print(f"Statistics saved to: {stats_path}")

        else:
            print("No data to process after filtering")
    else:
        print("No items with errors found")


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


def llm_scoring(
    input_path: Path,
    output_path: Path,
    lang: str,
    model_name: str = "qwen-3",
    batch_size: int = 1024,
    tensor_parallel_size: int = 1,
    compare_model: bool = False,
    model_max_length: int = 40960,
) -> None:
    """LLM-based code quality scoring using GPU processing"""
    pipeline = get_rewrite_pipeline(
        lang=lang, model_name=model_name, tensor_parallel_size=tensor_parallel_size, model_max_length=model_max_length
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

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for batch in stream_jsonl(input_path, batch_size):
            total_items += len(batch)
            print(f"Processing batch of {len(batch)} items...")

            # key in batch should be "text_formatted" for LLM scoring
            if not all("text_formatted" in item for item in batch):
                raise ValueError("All items in the batch must contain 'text_formatted' key for LLM scoring")
            codes = [item.get("text_formatted", "") for item in batch]

            # Call pipeline.score_codes
            try:
                evaluations = pipeline.get_scores(codes)
                scores = extract_scores_from_multiple_texts(evaluations)

                # Write results to output file
                for index, item in enumerate(batch):
                    score = scores[index] if index < len(scores) else 0  # Default to
                    item[score_key] = score
                    item[evaluation_key] = evaluations[index] if index < len(evaluations) else ""
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error during scoring: {e}")

    actual_time = time.time() - start_time
    print(f"LLM scoring completed: {actual_time:.1f}s total ({actual_time / total_items:.3f}s per item)")


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


# === CLI Entrypoint ===

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Code Quality Pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Auto-format subcommand
    p1 = sub.add_parser("auto_format", help="Auto-format code using CPU processing")
    p1.add_argument("--input-jsonl", type=Path, required=True)
    p1.add_argument("--output-jsonl", type=Path, required=True)
    p1.add_argument("--workers", type=int, default=32, help="Number of CPU workers")
    p1.add_argument("--lang", type=str, required=True, help="Programming language (e.g., python, rust, java)")
    p1.add_argument("--batch-size", type=int, default=1000, help="Batch size for CPU processing")
    p1.add_argument("--target-key", type=str, default="text", help="Key in JSON object to format (default: text)")

    # LLM auto-fix subcommand
    p2 = sub.add_parser("llm_auto_fix", help="LLM-based automatic bug fixing")
    p2.add_argument("--input-jsonl", type=Path, required=True)
    p2.add_argument("--output-dir", type=Path, required=True)
    p2.add_argument("--model", type=str, default="qwen-3", help="Local Qwen model identifier for vLLM")
    p2.add_argument("--lang", type=str, required=True, help="Programming language (e.g., python, rust, java)")
    p2.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    p2.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use for tensor parallelism")
    p2.add_argument("--model-max-length", type=int, default=40960, help="Maximum model length")

    # separate long context data
    p3 = sub.add_parser("long_context_sample", help="Separate code samples with and without linter errors")
    p3.add_argument("--input-jsonl", type=Path, required=True, help="Input JSONL file containing code samples")
    p3.add_argument("--output-path", type=Path, required=True, help="Output JSONL file path to save separated files")
    p3.add_argument("--tokenizer", type=Path)
    p3.add_argument("--threshold-length", type=int, default=20480, help="Threshold length for separating samples")

    # LLM scoring subcommand
    p4 = sub.add_parser("llm_scoring", help="LLM-based code quality scoring")
    p4.add_argument("--input-jsonl", type=Path, required=True)
    p4.add_argument("--output-jsonl", type=Path, required=True)
    p4.add_argument("--model", type=str, default="qwen-3", help="Local Qwen model identifier for vLLM")
    p4.add_argument("--lang", type=str, required=True, help="Programming language (e.g., python, rust, java)")
    p4.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    p4.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use for tensor parallelism")
    p4.add_argument("--compare-model", action="store_true", help="Compare with another model")
    p4.add_argument("--model-max-length", type=int, default=40960, help="Maximum model length for scoring")

    # LLM rewrite subcommand
    p5 = sub.add_parser("rewrite", help="LLM-based code rewriting")
    p5.add_argument("--input-jsonl", type=Path, required=True)
    p5.add_argument("--output-jsonl", type=Path, required=True)
    p5.add_argument("--lang", type=str, required=True, help="Programming language (e.g., python, rust, java)")
    p5.add_argument("--model", type=str, default="qwen-3", help="Local Qwen model identifier for vLLM")
    p5.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    p5.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use for tensor parallelism")
    p5.add_argument("--model-max-length", type=int, default=40960, help="Maximum model length for rewriting")
    p5.add_argument(
        "--prompt-type",
        type=str,
        default="stage5",
        choices=["stage5", "stage8"],
        help="Prompt type for rewriting: stage5 (first rewrite) or stage8 (second rewrite)",
    )

    # Competitive Programming LLM write subcommand
    p6 = sub.add_parser("competitive_programming_write", help="LLM-based code generation for competitive programming")
    p6.add_argument("--input-jsonl", type=Path, required=True)
    p6.add_argument("--output-jsonl", type=Path, required=True)
    p6.add_argument("--lang", type=str, required=True, help="Programming language (e.g., python, rust, java)")
    p6.add_argument("--model", type=str, default="qwen-3", help="Local Qwen model identifier for vLLM")
    p6.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    p6.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use for tensor parallelism")
    p6.add_argument("--model-max-length", type=int, default=40960, help="Maximum model length for rewriting")

    # format check
    p7 = sub.add_parser("format_check", help="Check if the input JSONL file is properly formatted")
    p7.add_argument("--input-jsonl", type=Path, required=True, help="Input JSONL file to check format")
    p7.add_argument("--output-jsonl", type=Path, required=True, help="Output JSONL file to save formatted items")
    p7.add_argument("--lang", type=str, required=True, help="Programming language (e.g., python, rust, java)")
    p7.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    p7.add_argument(
        "--target-key", type=str, default="improved_code", help="Key in JSON object to format (default: text)"
    )
    p7.add_argument("--workers", type=int, default=16, help="Number of CPU workers for formatting")

    # filter_rewritten_code
    p8 = sub.add_parser("filter_rewritten_code", help="Filter out error-containing data from JSONL files")
    p8.add_argument("--input-dir", type=Path, required=True, help="Input directory containing JSONL files")
    p8.add_argument("--output-dir", type=Path, required=True, help="Output directory to save filtered results")
    p8.add_argument(
        "--workers", type=int, default=None, help="Number of workers for parallel processing (default: CPU count)"
    )

    # 2nd rewrite
    p9 = sub.add_parser("second_rewrite", help="Second rewrite stage for code quality improvement")
    p9.add_argument("--input-jsonl", type=Path, required=True, help="Input JSONL file for second rewrite")
    p9.add_argument("--output-jsonl", type=Path, required=True, help="Output JSONL file for second rewrite")
    p9.add_argument("--lang", type=str, required=True, help="Programming language (e.g., python, rust, java)")
    p9.add_argument("--model", type=str, default="qwen-3", help="Local Qwen model identifier for vLLM")
    p9.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    p9.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use for tensor parallelism")
    p9.add_argument("--model-max-length", type=int, default=40960, help="Maximum model length for rewriting")
    p9.add_argument(
        "--prompt-type",
        type=str,
        default="stage8",
        choices=["stage5", "stage8"],
        help="Prompt type for rewriting: stage5 (first rewrite) or stage8 (second rewrite)",
    )

    # finemath rewrite
    p10 = sub.add_parser("finemath_rewrite", help="Finemath code rewriting")
    p10.add_argument("--input-jsonl", type=Path, required=True, help="Input JSONL file for finemath rewrite")
    p10.add_argument("--output-jsonl", type=Path, required=True, help="Output JSONL file for finemath rewrite")
    p10.add_argument("--model", type=str, default="qwen-3", help="Local Qwen model identifier for vLLM")
    p10.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    p10.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use for tensor parallelism")
    p10.add_argument("--model-max-length", type=int, default=40960, help="Maximum model length for rewriting")
    p10.add_argument(
        "--prompt-type",
        type=str,
        default="pre-train-text",
        choices=[
            "pre-train-text",
            "text-book-style",
            "question-answer",
            "planning-approach",
            "socratic-method",
            "multiple-solution",
        ],
        help="Prompt type for finemath rewriting",
    )

    # math difficulty scoring
    p11 = sub.add_parser("math_difficulty_scoring", help="Math difficulty scoring")
    p11.add_argument("--input-jsonl", type=Path, required=True, help="Input JSONL file for math difficulty scoring")
    p11.add_argument("--output-jsonl", type=Path, required=True, help="Output JSONL file for math difficulty scoring")
    p11.add_argument("--model", type=str, default="qwen-3", help="Local Qwen model identifier for vLLM")
    p11.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    p11.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use for tensor parallelism")
    p11.add_argument("--model-max-length", type=int, default=40960, help="Maximum model length for scoring")
    p11.add_argument(
        "--prompt-type",
        type=str,
        default="math_difficulty",
        choices=["math_difficulty"],
        help="Prompt type for math difficulty scoring",
    )

    args = parser.parse_args()

    if args.cmd == "auto_format":  # stage 1
        auto_format(
            input_path=args.input_jsonl,
            output_path=args.output_jsonl,
            n_workers=args.workers,
            lang=args.lang,
            batch_size=args.batch_size,
            target_key=args.target_key,
        )
    elif args.cmd == "llm_auto_fix":  # stage 2
        llm_auto_fix(
            input_path=args.input_jsonl,
            output_dir=args.output_dir,
            model_name=args.model,
            lang=args.lang,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            model_max_length=args.model_max_length,
        )
    elif args.cmd == "long_context_sample":  # stage 3
        separate_code_samples(
            input_path=args.input_jsonl,
            output_path=args.output_path,
            tokenizer_path=args.tokenizer,
            threshold_length=args.threshold_length,
        )
    elif args.cmd == "llm_scoring":  # stage 4
        llm_scoring(
            input_path=args.input_jsonl,
            output_path=args.output_jsonl,
            model_name=args.model,
            lang=args.lang,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            compare_model=args.compare_model,
            model_max_length=args.model_max_length,
        )
    elif args.cmd == "rewrite":  # stage 5
        llm_rewrite(
            input_path=args.input_jsonl,
            output_path=args.output_jsonl,
            lang=args.lang,
            model_name=args.model,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            model_max_length=args.model_max_length,
            prompt_type=args.prompt_type,
        )
    elif args.cmd == "competitive_programming_write":
        competitive_programming_write(
            input_path=args.input_jsonl,
            output_path=args.output_jsonl,
            lang=args.lang,
            model_name=args.model,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            model_max_length=args.model_max_length,
        )
    elif args.cmd == "format_check":  # stage 6, 9
        auto_format(
            input_path=args.input_jsonl,
            output_path=args.output_jsonl,
            n_workers=args.workers,
            lang=args.lang,
            batch_size=args.batch_size,
            target_key=args.target_key,
        )
    elif args.cmd == "filter_rewritten_code":  # stage 7, 10
        after_rewrite_filter(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            workers=args.workers,
        )
    elif args.cmd == "second_rewrite":  # stage 8
        llm_rewrite(
            input_path=args.input_jsonl,
            output_path=args.output_jsonl,
            lang=args.lang,
            model_name=args.model,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            model_max_length=args.model_max_length,
            prompt_type=args.prompt_type,
        )
    elif args.cmd == "finemath_rewrite":
        math_rewrite(
            input_path=args.input_jsonl,
            output_path=args.output_jsonl,
            model_name=args.model,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            model_max_length=args.model_max_length,
            prompt_type=args.prompt_type,
        )
    else:
        raise ValueError(f"Unknown command: {args.cmd}")
