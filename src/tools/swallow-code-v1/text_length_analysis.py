import argparse
import json
import os

parser = argparse.ArgumentParser(description="Calculate average length of values for a given key in JSONL files.")
parser.add_argument("--input-dir", required=True, help="Directory containing JSONL files")
parser.add_argument("--analysis-key", required=True, help="Key to analyze in the JSON objects")

args = parser.parse_args()

total_length = 0
count = 0

for filename in os.listdir(args.input_dir):
    if filename.endswith(".jsonl"):
        filepath = os.path.join(args.input_dir, filename)
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if args.analysis_key in data:
                        value = data[args.analysis_key]
                        if isinstance(value, str):
                            total_length += len(value)
                            count += 1
                        else:
                            print(
                                f"Warning: Value for key '{args.analysis_key}' in {filename} is not a string, skipping."
                            )

if count > 0:
    average_length = total_length / count
    print(f"Average length: {average_length}")
else:
    print("No valid values found.")
