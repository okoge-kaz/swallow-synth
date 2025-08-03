import json
import argparse
import glob
import os
from pathlib import Path
from tqdm import tqdm


def filter_jsonl_files(input_dir: str, output_dir: str) -> dict:
    """
    Process JSONL files matching '*low*.jsonl' in input_dir, filter records where score != 0,
    save to output_dir with the same filename, and calculate the proportion of non-zero scores.

    Args:
        input_dir: Directory containing input JSONL files.
        output_dir: Directory to save filtered JSONL files.

    Returns:
        Dictionary mapping file paths to proportion of records with non-zero scores.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all JSONL files matching '*low*.jsonl'
    input_files = glob.glob(os.path.join(input_dir, "*low*.json"))
    if not input_files:
        raise ValueError(f"No files matching '*low*.jsonl' found in {input_dir}")

    results = {}

    for input_file in input_files:
        file_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, file_name)

        total_count = 0
        non_zero_count = 0
        filtered_records = []

        # Read and process input file
        with open(input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Processing {file_name}"):
                try:
                    record = json.loads(line.strip())
                    total_count += 1
                    score = record.get("score", 0)
                    # Handle both int/float and string representations of numbers
                    try:
                        score = float(score)
                    except (TypeError, ValueError):
                        score = 0
                    if score != 0:
                        non_zero_count += 1
                        filtered_records.append(line.strip())
                except json.JSONDecodeError:
                    continue  # Skip invalid JSON lines

        # Write filtered records to output file
        if filtered_records:
            with open(output_file, "w", encoding="utf-8") as f:
                for record in filtered_records:
                    f.write(record + "\n")

        # Calculate proportion of non-zero scores
        proportion = non_zero_count / total_count if total_count > 0 else 0
        results[input_file] = proportion

    return results


def main():
    parser = argparse.ArgumentParser(description="Filter JSONL files with non-zero scores.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input JSONL files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save filtered JSONL files.")

    args = parser.parse_args()

    results = filter_jsonl_files(args.input_dir, args.output_dir)

    for file_path, proportion in results.items():
        print(f"File: {file_path}")
        print(f"Proportion of records with non-zero score: {proportion:.4%}")
        print()


if __name__ == "__main__":
    main()
