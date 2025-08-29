import argparse
import json
from tqdm import tqdm


def remove_duplicates(input_file, output_file):
    seen_titles = set()
    unique_lines = []

    # Count total lines for progress bar
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Read input JSONL file with progress bar
    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Processing JSONL"):
            try:
                data = json.loads(line.strip())
                title = data.get("title")
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_lines.append(line.strip())
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
                continue

    # Write to output JSONL file with progress bar
    with open(output_file, "w", encoding="utf-8") as f:
        for line in tqdm(unique_lines, desc="Writing output"):
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Remove duplicate entries based on title from a JSONL file.")
    parser.add_argument("--input-jsonl", required=True, help="Path to input JSONL file")
    parser.add_argument("--output-jsonl", required=True, help="Path to output JSONL file")
    args = parser.parse_args()

    remove_duplicates(args.input_jsonl, args.output_jsonl)


if __name__ == "__main__":
    main()
