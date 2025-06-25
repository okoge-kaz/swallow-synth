import argparse
import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


def load_source_datasets():
    print("Loading source datasets...")
    hf_datasets = {
        "taco": load_dataset("BAAI/TACO", trust_remote_code=True),
        "apps": load_dataset("codeparrot/apps", trust_remote_code=True),
        "code_contests": load_dataset("deepmind/code_contests"),
        "open-r1/codeforces": load_dataset("open-r1/codeforces"),
    }
    return hf_datasets


def get_question(hf_datasets, ds_name, split, index):
    try:
        benchmark = hf_datasets[ds_name][split][int(index)]

        if ds_name == "code_contests":
            if not benchmark["description"]:
                return None
            return benchmark["description"]
        elif ds_name in ["taco", "apps"]:
            return benchmark["question"]
        elif ds_name == "open-r1/codeforces":
            if not benchmark["description"]:
                return None
            question = benchmark["description"]
            if benchmark["input_format"]:
                question += "\n\nInput\n\n" + benchmark["input_format"]
            if benchmark["output_format"]:
                question += "\n\nOutput\n\n" + benchmark["output_format"]
            if benchmark["examples"]:
                question += "\n\nExamples"
                for example in benchmark["examples"]:
                    if "input" in example:
                        question += "\n\nInput\n\n" + example["input"]
                    if "output" in example:
                        question += "\n\nOutput\n\n" + example["output"]
            if benchmark["note"]:
                question += "\n\nNote\n\n" + benchmark["note"]
            return question
    except Exception as e:
        print(f"Error retrieving question for {ds_name}/{split}/{index}: {e}")
        return None

    return None


def process_and_save_dataset(output_dir, output_format="jsonl"):
    hf_datasets = load_source_datasets()

    print("Loading OpenCodeReasoning-2 dataset...")
    ocr2_dataset = load_dataset("nvidia/OpenCodeReasoning-2")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for lang in ["python", "cpp"]:
        print(f"\n{lang} dataset processing started...")
        ocr2_ds = ocr2_dataset[lang]  # type: ignore

        if output_format == "jsonl":
            output_file = output_path / f"ocr2_{lang}_with_questions.jsonl"
        else:  # json
            output_file = output_path / f"ocr2_{lang}_with_questions.json"

        processed_items = []
        success_count = 0
        error_count = 0

        for item in tqdm(ocr2_ds, desc=f"Processing {lang}"):
            processed_item = dict(item)

            ds_name = processed_item["dataset"]
            ds_split = processed_item["split"]
            ds_index = int(processed_item["index"])

            if ds_name not in ["taco", "apps", "code_contests", "open-r1/codeforces"]:
                print(f"Warning: unsupported dataset {ds_name} - {ds_split}/{ds_index}")
                error_count += 1
                continue

            question = get_question(hf_datasets, ds_name, ds_split, ds_index)

            if question is None:
                print(f"Warning: No question found for {ds_name} - {ds_split}/{ds_index}")
                error_count += 1
                processed_items.append(processed_item)
                continue

            if processed_item["question"] != "-":
                print(f"Warning: question already exists for {ds_name} - {ds_split}/{ds_index}")

            processed_item["question"] = question
            processed_items.append(processed_item)
            success_count += 1

        if output_format == "jsonl":
            with open(output_file, "w", encoding="utf-8") as f:
                for item in processed_items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:  # json
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(processed_items, f, ensure_ascii=False, indent=2)

        print(f"{lang} dataset processing completed!")
        print(f"  - Success: {success_count}")
        print(f"  - Errors: {error_count}")
        print(f"  - Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-dir", required=True, type=str)

    parser.add_argument("--format", choices=["jsonl", "json"], default="jsonl")

    args = parser.parse_args()

    try:
        process_and_save_dataset(args.output_dir, args.format)
        print("\nProcessing completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
