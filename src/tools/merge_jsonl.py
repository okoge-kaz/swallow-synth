import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Generator, List


def find_jsonl_files(directory: Path) -> List[Path]:
    """Find JSONL files in the specified directory"""
    jsonl_files = []
    for file_path in directory.rglob("*.jsonl"):
        if file_path.is_file():
            jsonl_files.append(file_path)
    return jsonl_files


def read_jsonl_file(file_path: Path) -> Generator[Dict[Any, Any], None, None]:
    """Read JSONL file and yield each line as a JSON object"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON at line {line_num} in {file_path}: {e}", file=sys.stderr)
                    continue
    except FileNotFoundError:
        print(f"Error: File {file_path} not found", file=sys.stderr)
    except PermissionError:
        print(f"Error: Permission denied to read file {file_path}", file=sys.stderr)


def check_key_length(data: Dict[Any, Any], key: str, min_length: int = 10) -> bool:
    """Check if the specified key's value has minimum length"""
    if key not in data:
        return False

    value = data[key]
    if not isinstance(value, str):
        return False

    return len(value) >= min_length


def merge_jsonl_files(input_dir: Path, output_file: Path, key: str, min_length: int = 10) -> None:
    """Merge JSONL files and write to output"""
    jsonl_files = find_jsonl_files(input_dir)

    if not jsonl_files:
        print(f"Warning: No JSONL files found in {input_dir}", file=sys.stderr)
        return

    print(f"Found JSONL files: {len(jsonl_files)}")
    for file_path in jsonl_files:
        print(f"  - {file_path}")

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    valid_lines = 0

    try:
        with open(output_file, "w", encoding="utf-8") as out_f:
            for file_path in jsonl_files:
                print(f"Processing: {file_path}")
                file_lines = 0
                file_valid_lines = 0

                for data in read_jsonl_file(file_path):
                    total_lines += 1
                    file_lines += 1

                    if check_key_length(data, key, min_length):
                        json.dump(data, out_f, ensure_ascii=False)
                        out_f.write("\n")
                        valid_lines += 1
                        file_valid_lines += 1

                print(f"  - Total lines: {file_lines}, Valid lines: {file_valid_lines}")

        print(f"\nProcessing completed!")
        print(f"Total processed lines: {total_lines}")
        print(f"Valid lines: {valid_lines}")
        print(f"Output file: {output_file}")

    except PermissionError:
        print(f"Error: Permission denied to write to {output_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error during processing: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Merge JSONL files and extract only lines where the specified key's value has the minimum length."
    )

    parser.add_argument("--input-dir", required=True, type=Path, help="Path to the directory containing JSONL files")

    parser.add_argument("--output-jsonl", required=True, type=Path, help="Path to the output JSONL file")

    parser.add_argument("--key", default="generated_text", help="Key name to check (default: generated_text)")

    parser.add_argument("--min-length", type=int, default=10, help="Minimum length of the key's value (default: 10)")

    args = parser.parse_args()

    # Check if input directory exists
    if not args.input_dir.exists():
        print(f"Error: Input directory {args.input_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output_jsonl}")
    print(f"Target key: {args.key}")
    print(f"Minimum length: {args.min_length}")
    print("-" * 50)

    merge_jsonl_files(args.input_dir, args.output_jsonl, args.key, args.min_length)


if __name__ == "__main__":
    main()
