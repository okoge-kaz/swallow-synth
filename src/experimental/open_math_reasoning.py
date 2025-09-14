import argparse
import json
from pathlib import Path


def process_jsonl_files(input_directory: str, output_file: str) -> None:
    input_path = Path(input_directory)

    if not input_path.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_directory}")

    if not input_path.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {input_directory}")

    jsonl_files = list(input_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"Warning: No JSONL files found in the directory: {input_directory}")
        return

    print(f"Found {len(jsonl_files)} JSONL files in the directory: {input_directory}")

    processed_count = 0

    with open(output_file, "w", encoding="utf-8") as outfile:
        for jsonl_file in jsonl_files:
            print(f"Processing: {jsonl_file.name}")

            try:
                with open(jsonl_file, "r", encoding="utf-8") as infile:
                    for line_num, line in enumerate(infile, 1):
                        line = line.strip()

                        if not line:
                            continue

                        try:
                            data = json.loads(line)

                            required_fields = ["problem", "generated_solution", "expected_answer"]
                            missing_fields = [field for field in required_fields if field not in data]

                            if missing_fields:
                                print(
                                    f"Warning: {jsonl_file.name}:{line_num} - Missing fields: {', '.join(missing_fields)}"
                                )
                                continue

                            text_content = (
                                "Problem:\n\n"
                                + str(data["problem"])
                                + "\n\n"
                                + "Solution:\n\n"
                                + str(data["generated_solution"])
                                + "\n\n"
                                + "Answer:\n\n"
                                + str(data["expected_answer"])
                            )

                            output_data = data.copy()
                            output_data["text"] = text_content

                            outfile.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                            processed_count += 1

                        except json.JSONDecodeError as e:
                            print(f"Error: {jsonl_file.name}:{line_num} - JSON decode error: {e}")
                            continue

            except Exception as e:
                print(f"Error processing file {jsonl_file.name}: {e}")
                continue

    print(f"Processing complete. {processed_count} records written to {output_file}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--directory", type=str, required=True, help="input directory containing JSONL files")

    parser.add_argument("--output-jsonl", required=True, help="output JSONL file path")

    args = parser.parse_args()

    try:
        process_jsonl_files(args.directory, args.output_jsonl)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
