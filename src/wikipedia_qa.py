import argparse
import json
import os
from typing import cast, Dict, List, Union
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, PreTrainedTokenizer


ENGLISH_MULTI_CHOICE_PROMPT = """Generate a multiple-choice question with 4 options and the correct answer based on the following text:

{text}

Use the following format and style, as shown in the examples below:

Example:
Question: What is the primary pigment involved in photosynthesis that absorbs light in the blue and red wavelengths?
Options:
A. Chlorophyll
B. Carotene
C. Xanthophyll
D. Phycocyanin
Answer: A

Output the one pair of question, options, and answer in the same format as the examples.
"""

ENGLISH_FREE_STYLE_PROMPT = """Generate an open-ended question and its answer based on the following text:

{text}

Use the following format and style, as shown in the examples below:

Example:
Question: How does the absorption spectrum of chlorophyll influence the efficiency of photosynthesis in plants?
Answer: The absorption spectrum of chlorophyll, which primarily absorbs blue and red wavelengths while reflecting green, significantly influences photosynthesis efficiency. By absorbing light in these wavelengths, chlorophyll optimizes energy capture from sunlight, as blue and red light provide sufficient energy to excite electrons in the photosystem. However, the reflection of green light reduces efficiency, as some usable light energy is lost. Plants adapt to specific light environments, with accessory pigments like carotenoids capturing additional wavelengths, enhancing overall efficiency.

Output the one pair of question and answer in the same format as the examples.
"""


JAPANESE_MULTI_CHOICE_PROMPT = """以下のテキストに基づいて、4つの選択肢を持つ多肢選択式の質問と正解を1組だけ作成してください:

{text}

以下の形式で質問と正解を1組だけ出力してください。

例:
Question: 光合成において、青と赤の波長の光を吸収する主要な色素は何ですか？
Options:
A. クロロフィル
B. カロテン
C. キサントフィル
D. フィコシアニン
Answer: A

例と同じフォーマットで質問、選択肢、正解を1組だけ出力してください。
"""

JAPANESE_FREE_STYLE_PROMPT = """以下のテキストに基づいて、自由記述式の質問とその答えを作成してください:

{text}

以下のフォーマットとスタイルで、例に従ってください:

例:
Question: クロロフィルの吸収スペクトルは、植物の光合成の効率にどのように影響しますか？
Answer: クロロフィルの吸収スペクトルは、主に青と赤の波長を吸収し、緑を反射するため、光合成の効率に大きな影響を与えます。これらの波長の光を吸収することで、クロロフィルは光化学系で電子を励起するのに十分なエネルギーを最適に捕捉します。しかし、緑色の光の反射により、一部の使用可能な光エネルギーが失われるため、効率が低下します。植物は、カロテノイドなどの補助色素が追加の波長を捕捉することで、特定の光環境に適応し、全体的な効率を高めています。

例と同じフォーマットで質問と答えを1ペア出力してください。
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with clear defaults and type hints."""
    parser = argparse.ArgumentParser(description="Generate QA from JSONL using vLLM.")
    parser.add_argument("--input-jsonl", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument("--output-jsonl", type=str, required=True, help="Path to output JSONL file.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the vLLM model.")
    parser.add_argument(
        "--qa-mode",
        type=str,
        choices=["choices", "free"],
        default="choices",
        help="QA mode: 'choices' for multiple-choice, 'free' for open-ended.",
    )
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for processing and saving.")
    parser.add_argument(
        "--lang",
        type=str,
        choices=["ja", "en"],
        default="ja",
        help="Prompt language: 'ja' (Japanese) or 'en' (English).",
    )
    parser.add_argument("--chunk-max-tokens", type=int, default=1024, help="Max tokens per text chunk.")
    parser.add_argument("--gen-max-tokens", type=int, default=16384, help="Max tokens for generated output.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for vLLM.")
    return parser.parse_args()


def load_tokenizer(model_path: str) -> PreTrainedTokenizer:
    """Load and return a tokenizer for the specified model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def get_prompt_template(lang: str, qa_mode: str) -> str:
    """Return the appropriate prompt template based on language and QA mode."""
    prompt_map = {
        ("en", "choices"): ENGLISH_MULTI_CHOICE_PROMPT,
        ("en", "free"): ENGLISH_FREE_STYLE_PROMPT,
        ("ja", "choices"): JAPANESE_MULTI_CHOICE_PROMPT,
        ("ja", "free"): JAPANESE_FREE_STYLE_PROMPT,
    }
    return prompt_map.get((lang, qa_mode), ENGLISH_MULTI_CHOICE_PROMPT)


def read_jsonl(file_path: str) -> List[Dict[str, Union[str, List]]]:
    """Read and parse JSONL file into a list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_to_jsonl(file_path: str, data_list: List[Dict[str, Union[str, List]]]) -> None:
    """Append data to a JSONL file."""
    mode = "a" if os.path.exists(file_path) else "w"
    with open(file_path, mode, encoding="utf-8") as f:
        for entry in data_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def parse_generated_text(text: str, qa_mode: str) -> Dict[str, Union[str, List[str]]]:
    """Parse generated text into a structured QA dictionary."""
    print(f"Generated Text:\n{text}\n{'-' * 40}", flush=True)
    lines = text.split("\n")
    qa: Dict[str, Union[str, List[str]]] = {}

    try:
        if qa_mode == "choices":
            qa["question"] = next((line.split("Question: ")[1] for line in lines if line.startswith("Question: ")), "")
            qa["options"] = [line for line in lines if line.startswith(("A.", "B.", "C.", "D."))]
            qa["answer"] = next((line.split("Answer: ")[1] for line in lines if line.startswith("Answer: ")), "")
        else:
            qa["question"] = next((line.split("Question: ")[1] for line in lines if line.startswith("Question: ")), "")
            qa["answer"] = next((line.split("Answer: ")[1] for line in lines if line.startswith("Answer: ")), "")
    except Exception as e:
        qa = {"error": f"Failed to parse {qa_mode} QA: {str(e)}"}

    return qa


def process_batch(
    batch_lines: List[Dict[str, Union[str, List]]],
    tokenizer: PreTrainedTokenizer,
    llm: LLM,
    prompt_template: str,
    chunk_max_tokens: int,
    gen_max_tokens: int,
    qa_mode: str,
) -> List[Dict[str, Union[str, List]]]:
    """Process a batch of JSONL lines and generate QA."""
    batch_inputs = []
    num_qa_per_line: Dict[int, int] = {}

    for local_idx, item in enumerate(batch_lines):
        concat_text = ""
        concat_text_tokens = 0

        for section in item.get("sections", []):
            section = cast(Dict[str, str], section)
            section_text = section.get("text", "")
            tokens = tokenizer.encode(section_text, add_special_tokens=False)

            concat_text += section_text
            concat_text_tokens += len(tokens)

            if concat_text_tokens > chunk_max_tokens:
                batch_inputs.append(prompt_template.format(text=concat_text))
                num_qa_per_line[local_idx] = num_qa_per_line.get(local_idx, 0) + 1
                concat_text = ""
                concat_text_tokens = 0

        if concat_text:
            batch_inputs.append(prompt_template.format(text=concat_text))
            num_qa_per_line[local_idx] = num_qa_per_line.get(local_idx, 0) + 1

    # Generate QA using vLLM
    outputs = llm.generate(
        batch_inputs,
        sampling_params=SamplingParams(
            max_tokens=gen_max_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=["\n\n"],
        ),
    )

    qas = [parse_generated_text(output.outputs[0].text, qa_mode) for output in outputs]
    processed_lines = batch_lines.copy()

    # Distribute generated QAs to corresponding lines
    for line_idx, num_qa in num_qa_per_line.items():
        item = processed_lines[line_idx]
        item_qas = []
        for _ in range(num_qa):
            if qas:
                qa = qas.pop(0)
                item_qas.append(qa)
                if "error" in qa:
                    print(f"Warning: {qa['error']} in line {line_idx}")
                else:
                    if qa_mode == "choices":
                        item["text"] += (
                            f"\n\nQuestion: {qa['question']}\nOptions:\n"
                            + "\n".join(qa["options"])
                            + f"\nAnswer: {qa['answer']}\n"
                        )
                    else:
                        item["text"] += f"\n\nQuestion: {qa['question']}\nAnswer: {qa['answer']}\n"
        item["question_answers"] = item_qas

    return processed_lines


def main():
    """Main function to process JSONL and generate QA using vLLM."""
    args = parse_args()

    # Initialize tokenizer and vLLM
    tokenizer = load_tokenizer(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size)

    # Get prompt template
    prompt_template = get_prompt_template(args.lang, args.qa_mode)

    # Read input JSONL
    lines = read_jsonl(args.input_jsonl)

    # Process in batches
    for start_idx in range(0, len(lines), args.batch_size):
        batch_lines = lines[start_idx : start_idx + args.batch_size]
        processed_lines = process_batch(
            batch_lines, tokenizer, llm, prompt_template, args.chunk_max_tokens, args.gen_max_tokens, args.qa_mode
        )
        save_to_jsonl(args.output_jsonl, processed_lines)


if __name__ == "__main__":
    main()
