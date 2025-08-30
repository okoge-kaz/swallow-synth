import argparse
import json
import os
from multiprocessing import Pool, cpu_count
import re


def extract_and_format_text(raw_text):
    """
    Extract and format the text based on the rules.
    """
    # Check for <|MATH_TEXT|>
    math_marker = "<|MATH_TEXT|>"
    if math_marker in raw_text:
        return raw_text.rsplit(math_marker, 1)[-1].strip()

    # Check for <think> and </think>
    think_start = "<think>"
    think_end = "</think>"

    # Find the position of the first </think>
    end_pos = raw_text.find(think_end)
    if end_pos != -1:
        # Delete everything up to and including </think>
        formatted = raw_text[end_pos + len(think_end) :].strip()
    else:
        formatted = raw_text.strip()

    # Check if text length is at least 50 characters
    if len(formatted) < 50:
        return ""  # Discard if too short

    # Discard if <think> or </think> remains in the formatted text
    if think_start in formatted or think_end in formatted:
        return ""

    return formatted


def has_repetitions(text):
    """
    Detect if the text contains a substring repeated 10+ times consecutively.
    Returns True if repetitions are found, False otherwise.
    """
    # Use regex to find any group that repeats 10 times total (1 + 9 or more)
    if re.search(r"(.+?)\1{9,}", text):
        return True
    return False


def has_but_wait(text):
    """
    Check if the text contains 'But wait' (case-sensitive).
    Returns True if found, False otherwise.
    """
    return "But wait" in text


def process_line(line):
    """
    Process a single JSON line.
    """
    try:
        data = json.loads(line)
        raw_text = data.get("llm_extracted_math_text", "")
        formatted = extract_and_format_text(raw_text)
        # Discard text with repetitions, containing 'But wait', or containing <think> or </think>
        if formatted and not has_repetitions(formatted) and not has_but_wait(formatted):
            return {"text": formatted}
        return None
    except json.JSONDecodeError:
        return None


def process_file(input_path, output_path):
    """
    Process a single JSONL file.
    """
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            processed = process_line(line.strip())
            if processed:
                outfile.write(json.dumps(processed) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Process JSONL files for math text extraction and cleaning.")
    parser.add_argument("--input-dir", type=str, help="Input directory containing JSONL files.")
    parser.add_argument("--input-jsonl", type=str, help="Single input JSONL file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for processed files.")

    args = parser.parse_args()

    if args.input_dir and args.input_jsonl:
        raise ValueError("Specify either --input-dir or --input-jsonl, not both.")
    if not args.input_dir and not args.input_jsonl:
        raise ValueError("Specify either --input-dir or --input-jsonl.")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_jsonl:
        # Single file mode
        filename = os.path.basename(args.input_jsonl)
        output_path = os.path.join(args.output_dir, filename)
        process_file(args.input_jsonl, output_path)
    elif args.input_dir:
        # Directory mode with multiprocessing
        files = [f for f in os.listdir(args.input_dir) if f.endswith(".jsonl")]
        if not files:
            print("No JSONL files found in input directory.")
            return

        # Limit processes to min(cpu_count, len(files))
        num_processes = min(cpu_count(), len(files))

        # Prepare arguments for pool
        tasks = []
        for filename in files:
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, filename)
            tasks.append((input_path, output_path))

        with Pool(processes=num_processes) as pool:
            pool.starmap(process_file, tasks)


if __name__ == "__main__":
    main()
