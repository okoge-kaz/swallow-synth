import argparse
import json
import os
from pathlib import Path
from typing import Any


def process_jsonl_file(file_path: str):
    processed_data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                if not check_conditions(data):
                    continue

                processed_entry = process_data_entry(data)
                processed_data.append(processed_entry)

            except json.JSONDecodeError as e:
                print(f"JSON decode error in {file_path}, line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"Missing key in {file_path}, line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing {file_path}, line {line_num}: {e}")
                continue

    return processed_data


def check_conditions(data: dict[str, str | dict | float]) -> bool:
    try:
        if len(data.get("input")) <= 10 or len(data.get("output")) <= 10:  # type: ignore
            return False

        return True

    except (TypeError, AttributeError):
        print("Error checking conditions: data structure is not as expected.", flush=True)
        return False


def process_data_entry(data: dict[str, Any]) -> dict[str, Any]:
    processed_data = data.copy()

    input_text = data.get("input", "")
    output_text = data.get("output", "")
    processed_data["text"] = f"{input_text}\n\n{output_text}"

    return processed_data


def save_processed_data(processed_data: list, output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in processed_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-file",
        type=str,
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: specified input directory does not exist: {input_dir}")
        return
    if not input_dir.is_dir():
        print(f"Error: specified input path is not a directory: {input_dir}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = list(input_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"Warning: No JSONL files found in the input directory: {input_dir}")
        return

    print(f"Number of JSONL files found: {len(jsonl_files)}")

    all_processed_data = []
    total_processed = 0

    for jsonl_file in jsonl_files:
        print(f"Processing: {jsonl_file.name}")

        processed_data = process_jsonl_file(jsonl_file)  # type: ignore
        all_processed_data.extend(processed_data)
        total_processed += len(processed_data)

        print(f"  {jsonl_file.name}: the number of processed entries: {len(processed_data)}")

    output_file = output_dir / args.output_file
    save_processed_data(all_processed_data, output_file)

    print(f"\nProcessing complete!")
    print(f"  Total processed entries: {total_processed}")
    print(f"  Output saved to: {output_file}")


if __name__ == "__main__":
    main()
