import json
import random
import re
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from collections import defaultdict


def analyze_jsonl_files(
    file_paths: List[str], keys: List[str], sample_size: int, keywords: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze multiple JSONL files, sampling from each and estimating the proportion of records
    where specified keys contain each keyword (case-insensitive).

    Args:
        file_paths: List of paths to JSONL files.
        keys: List of keys to check for keywords in each file (same length as file_paths).
        sample_size: Number of records to sample from each file (default: 100,000).
        keywords: List of keywords to check (default: ['django', 'flask', 'torch', 'sklearn', 'matplotlib', 'keras', 'beautifulsoup', 'tensorflow']).

    Returns:
        Dictionary mapping file paths to another dict of keyword to estimated proportions.
    """
    if len(file_paths) != len(keys):
        raise ValueError("The number of file paths must match the number of keys.")

    if keywords is None:
        keywords = ["django", "flask", "torch", "sklearn", "matplotlib", "keras", "beautifulsoup", "tensorflow"]

    results = {}

    for file_path, key in zip(file_paths, keys):
        # Count total lines to determine sampling strategy
        total_lines = sum(1 for _ in open(file_path, "r", encoding="utf-8"))

        # If file has fewer lines than sample_size, use all lines
        if total_lines <= sample_size:
            indices = set(range(total_lines))
            sample_count = total_lines
        else:
            indices = set(random.sample(range(total_lines), sample_size))
            sample_count = sample_size

        keyword_counts = defaultdict(int)
        with open(file_path, "r", encoding="utf-8") as f:
            # Use tqdm for progress bar, but since we're skipping lines, adjust desc
            pbar = tqdm(total=sample_count, desc=f"Processing {Path(file_path).name}")
            for i, line in enumerate(f):
                if i in indices:
                    try:
                        record = json.loads(line.strip())
                        value = record.get(key, "").lower()
                        if isinstance(value, str):
                            for kw in keywords:
                                if re.search(re.escape(kw.lower()), value):
                                    keyword_counts[kw] += 1
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON lines
                    pbar.update(1)
            pbar.close()

        proportions = {kw: keyword_counts[kw] / sample_count if sample_count else 0 for kw in keywords}
        results[file_path] = proportions

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze JSONL files for keyword occurrences in specified keys.")
    parser.add_argument("--file-paths", type=str, nargs="+", required=True, help="List of JSONL file paths.")
    parser.add_argument("--keys", type=str, nargs="+", required=True, help="List of keys corresponding to each file.")
    parser.add_argument("--sample_size", type=int, default=100000, help="Number of samples per file (default: 100000).")
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        default=["django", "flask", "torch", "sklearn", "matplotlib", "keras", "BeautifulSoup", "tensorflow"],
        help="List of keywords to check (default: django flask torch sklearn matplotlib keras beautifulsoup tensorflow).",
    )

    args = parser.parse_args()

    if len(args.file_paths) != len(args.keys):
        parser.error("The number of file_paths must match the number of keys.")

    results = analyze_jsonl_files(args.file_paths, args.keys, args.sample_size, args.keywords)

    for file_path, proportions in results.items():
        print(f"File: {file_path}")
        analysis_key = args.keys[args.file_paths.index(file_path)]
        for kw, proportion in proportions.items():
            print(f"Estimated proportion of records with '{kw}' in key '{analysis_key}': {proportion:.4%}")
        print()


if __name__ == "__main__":
    main()
