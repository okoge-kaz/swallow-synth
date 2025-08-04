import json
import argparse
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Tuple, Optional
from transformers import AutoTokenizer


def process_file_math_filter(args: Tuple[Path, Path, str, int]) -> Dict[str, Any]:
    """Process a single file and filter based on math token length"""
    input_file, output_dir, tokenizer_path, max_tokens = args

    # Load tokenizer for this worker
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    filtered_items = []
    file_stats = {
        "total_items": 0,
        "items_kept": 0,
        "items_filtered": 0,
        "avg_tokens_kept": 0.0,
        "avg_tokens_filtered": 0.0,
        "max_tokens_seen": 0,
    }

    print(f"Processing {input_file.name}...")

    total_tokens_kept = 0
    total_tokens_filtered = 0

    with input_file.open("r", encoding="utf-8") as fin:
        for line in fin:
            item = json.loads(line)
            file_stats["total_items"] += 1

            # Get text content for token counting
            text = item.get("text", "")
            if not text:
                # Skip items without text content
                file_stats["items_filtered"] += 1
                continue

            # Count tokens
            token_count = len(tokenizer.encode(text))
            file_stats["max_tokens_seen"] = max(file_stats["max_tokens_seen"], token_count)

            if token_count <= max_tokens:
                # Keep item
                filtered_items.append(item)
                file_stats["items_kept"] += 1
                total_tokens_kept += token_count
            else:
                # Filter out item
                file_stats["items_filtered"] += 1
                total_tokens_filtered += token_count

    # Calculate averages
    if file_stats["items_kept"] > 0:
        file_stats["avg_tokens_kept"] = total_tokens_kept / file_stats["items_kept"]
    if file_stats["items_filtered"] > 0:
        file_stats["avg_tokens_filtered"] = total_tokens_filtered / file_stats["items_filtered"]

    # Write filtered results
    file_stem = input_file.stem
    output_file = output_dir / f"{file_stem}.jsonl"

    with output_file.open("w", encoding="utf-8") as fout:
        for item in filtered_items:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {"input_file": input_file.name, "output_file": output_file.name, "stats": file_stats}


def math_length_filter(
    input_dir: Path,
    output_dir: Path,
    tokenizer_path: str,
    max_tokens: int = 20480,
    workers: Optional[int] = None,
) -> None:
    """
    Filter math JSONL files based on token length with multiprocessing.

    Args:
        input_dir: Directory containing input JSONL files
        output_dir: Directory to save filtered JSONL files
        tokenizer_path: HuggingFace tokenizer model path
        max_tokens: Maximum token count allowed (default: 20480)
        workers: Number of worker processes (default: CPU count)
    """

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSONL files
    jsonl_files = list(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL files to process")
    print(f"Token limit: {max_tokens}")
    print(f"Tokenizer: {tokenizer_path}")

    # Set workers
    if workers is None:
        workers = cpu_count()

    # Limit workers to not exceed file count
    workers = min(workers, len(jsonl_files))
    print(f"Using {workers} workers for parallel processing")

    # Prepare arguments for multiprocessing
    args_list = [(file_path, output_dir, tokenizer_path, max_tokens) for file_path in jsonl_files]

    # Process files in parallel
    start_time = time.time()

    with Pool(workers) as pool:
        results = pool.map(process_file_math_filter, args_list)

    # Collect and display statistics
    total_stats = {
        "total_files": len(results),
        "total_items": 0,
        "items_kept": 0,
        "items_filtered": 0,
        "avg_tokens_kept": 0.0,
        "avg_tokens_filtered": 0.0,
        "max_tokens_seen": 0,
    }

    total_tokens_kept_sum = 0
    total_tokens_filtered_sum = 0

    print("\nPer-file results:")
    for result in results:
        stats = result["stats"]
        print(
            f"  {result['input_file']:<30} -> {result['output_file']:<35} "
            f"({stats['items_kept']:>6} kept, {stats['items_filtered']:>6} filtered, "
            f"max tokens: {stats['max_tokens_seen']:>6})"
        )

        # Accumulate totals
        total_stats["total_items"] += stats["total_items"]
        total_stats["items_kept"] += stats["items_kept"]
        total_stats["items_filtered"] += stats["items_filtered"]
        total_stats["max_tokens_seen"] = max(total_stats["max_tokens_seen"], stats["max_tokens_seen"])

        total_tokens_kept_sum += stats["avg_tokens_kept"] * stats["items_kept"]
        total_tokens_filtered_sum += stats["avg_tokens_filtered"] * stats["items_filtered"]

    # Calculate overall averages
    if total_stats["items_kept"] > 0:
        total_stats["avg_tokens_kept"] = total_tokens_kept_sum / total_stats["items_kept"]
    if total_stats["items_filtered"] > 0:
        total_stats["avg_tokens_filtered"] = total_tokens_filtered_sum / total_stats["items_filtered"]

    # Display final statistics
    processing_time = time.time() - start_time

    print(f"\nMath token length filtering completed in {processing_time:.1f}s:")
    print(f"  Total files processed: {total_stats['total_files']}")
    print(f"  Total items processed: {total_stats['total_items']}")
    print(
        f"  Items kept (≤{max_tokens} tokens): {total_stats['items_kept']} ({total_stats['items_kept'] / total_stats['total_items'] * 100:.1f}%)"
    )
    print(
        f"  Items filtered (>{max_tokens} tokens): {total_stats['items_filtered']} ({total_stats['items_filtered'] / total_stats['total_items'] * 100:.1f}%)"
    )
    print(f"  Average tokens (kept items): {total_stats['avg_tokens_kept']:.1f}")
    print(f"  Average tokens (filtered items): {total_stats['avg_tokens_filtered']:.1f}")
    print(f"  Maximum tokens encountered: {total_stats['max_tokens_seen']}")
    print(f"  Filtered files saved to: {output_dir}")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Filter math JSONL files based on token length",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing input JSONL files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save filtered JSONL files")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen3-32B-FP8",
        help="HuggingFace tokenizer model path",
    )
    parser.add_argument("--max-tokens", type=int, default=20480, help="Maximum token count allowed (default: 20480)")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: CPU count)")

    args = parser.parse_args()

    print("Math Token Length Filter Tool")
    print("=" * 50)

    math_length_filter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer,
        max_tokens=args.max_tokens,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
