import argparse
import json
import os
from tqdm import tqdm


def split_jsonl(input_file, output_dir, samples_per_file=100000):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Count total lines for progress bar
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    current_file_num = 1
    current_lines = []
    line_count = 0

    # Read input JSONL file with progress bar
    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Processing JSONL"):
            try:
                # Validate JSON
                json.loads(line.strip())
                current_lines.append(line.strip())
                line_count += 1

                # Write to file when reaching samples_per_file or at the end
                if line_count >= samples_per_file:
                    output_file = os.path.join(output_dir, f"en_wikipedia_{current_file_num}.jsonl")
                    with open(output_file, "w", encoding="utf-8") as out_f:
                        for out_line in tqdm(current_lines, desc=f"Writing en_wikipedia_{current_file_num}.jsonl"):
                            out_f.write(out_line + "\n")
                    current_file_num += 1
                    current_lines = []
                    line_count = 0

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
                continue

    # Write remaining lines to a final file if any
    if current_lines:
        output_file = os.path.join(output_dir, f"en_wikipedia_{current_file_num}.jsonl")
        with open(output_file, "w", encoding="utf-8") as out_f:
            for out_line in tqdm(current_lines, desc=f"Writing en_wikipedia_{current_file_num}.jsonl"):
                out_f.write(out_line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Split a JSONL file into multiple files with specified sample size.")
    parser.add_argument("--input-jsonl", required=True, help="Path to input JSONL file")
    parser.add_argument("--output-dir", required=True, help="Directory to save split JSONL files")
    args = parser.parse_args()

    split_jsonl(args.input_jsonl, args.output_dir)


if __name__ == "__main__":
    main()
