import argparse
import json
import os
from pathlib import Path


def split_jsonl_by_score(input_dir, output_dir):
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process each file matching the pattern
    for file_path in Path(input_dir).glob("train_*_Qwen3-14B.jsonl"):
        # Initialize output files
        base_name = file_path.stem
        low_file = Path(output_dir) / f"{base_name}_low_Qwen3-14B.json"
        medium_file = Path(output_dir) / f"{base_name}_medium_Qwen3-14B.json"
        high_file = Path(output_dir) / f"{base_name}_high_Qwen3-14B.json"

        with (
            open(low_file, "w", encoding="utf-8") as low_f,
            open(medium_file, "w", encoding="utf-8") as medium_f,
            open(high_file, "w", encoding="utf-8") as high_f,
        ):
            # Read input file
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        score = data.get("score", 0)

                        # Determine which file to write to based on score
                        if 0 <= score <= 2:
                            low_f.write(json.dumps(data) + "\n")
                        elif 3 <= score <= 6:
                            medium_f.write(json.dumps(data) + "\n")
                        elif 7 <= score <= 10:
                            high_f.write(json.dumps(data) + "\n")
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line in {file_path}")
                        continue


def main():
    parser = argparse.ArgumentParser(description="Split JSONL files by score ranges.")
    parser.add_argument("--input-dir", required=True, help="Input directory containing JSONL files")
    parser.add_argument("--output-dir", required=True, help="Output directory for split files")
    args = parser.parse_args()

    split_jsonl_by_score(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
