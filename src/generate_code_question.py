import argparse
import json
import os
from typing import Dict, List, Union
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, PreTrainedTokenizer

# Define the English prompt as a message structure for chat templates
ENGLISH_CHAT_MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful assistant tasked with generating problem statements for Python code snippets.",
    },
    {
        "role": "user",
        "content": """Given the following Python code as the answer, create a problem statement (Question) in English that describes what the code implements. The question should ask to implement Python code that achieves a specific functionality. Include any key conditions, behaviors, value ranges, constraints, input/output formats, edge cases, time/space complexity requirements.

Make the question detailed and self-contained so that the provided code would be a correct solution to it.

Code:
{code}

Please provide the question right after "Question: ".""",
    },
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with clear defaults and type hints."""
    parser = argparse.ArgumentParser(
        description="Generate problem statements (Questions) from Python code snippets in JSONL using vLLM."
    )
    parser.add_argument("--input-jsonl", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument("--output-jsonl", type=str, required=True, help="Path to output JSONL file.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the vLLM model.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for processing and saving.")
    parser.add_argument(
        "--lang",
        type=str,
        choices=["en"],
        default="en",
        help="Prompt language: 'en' (English). Output Question is always in English.",
    )
    parser.add_argument("--gen-max-tokens", type=int, default=16384, help="Max tokens for generated output.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for vLLM.")
    return parser.parse_args()


def load_tokenizer(model_path: str) -> PreTrainedTokenizer:
    """Load and return a tokenizer for the specified model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def get_prompt_template(lang: str) -> List[Dict[str, str]]:
    """Return the appropriate chat message template based on language."""
    prompt_map = {
        "en": ENGLISH_CHAT_MESSAGES,
    }
    return prompt_map.get(lang, ENGLISH_CHAT_MESSAGES)


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


def parse_generated_text(text: str) -> Dict[str, str]:
    """Parse generated text into a structured question dictionary."""
    print(f"Generated Text:\n{text}\n{'-' * 40}", flush=True)
    qa: Dict[str, str] = {}

    try:
        # Find the index of "Question: " in the text
        question_marker = "Question: "
        start_idx = text.find(question_marker)
        if start_idx == -1:
            raise ValueError("No 'Question: ' marker found in generated text")

        # Extract everything after "Question: " as the question
        question_text = text[start_idx + len(question_marker) :].strip()
        if not question_text:
            raise ValueError("Question text is empty")

        qa["question"] = question_text
    except Exception as e:
        qa = {"error": f"Failed to parse question: {str(e)}"}

    return qa


def process_batch(
    batch_lines: List[Dict[str, Union[str, List]]],
    tokenizer: PreTrainedTokenizer,
    llm: LLM,
    prompt_template: List[Dict[str, str]],
    gen_max_tokens: int,
) -> List[Dict[str, Union[str, List]]]:
    """Process a batch of JSONL lines and generate questions from code snippets using chat template."""
    batch_inputs = []
    valid_indices = []

    for local_idx, item in enumerate(batch_lines):
        code = item.get("improved_code", "")
        if code:
            # Create a copy of the prompt template and format the user message with the code
            formatted_messages = [msg.copy() for msg in prompt_template]
            for msg in formatted_messages:
                if msg["role"] == "user":
                    msg["content"] = msg["content"].format(code=code.strip())
            # Apply the chat template to format the prompt
            prompt = tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
            batch_inputs.append(prompt)
            valid_indices.append(local_idx)

    if not batch_inputs:
        return []

    # Generate questions using vLLM
    outputs = llm.generate(
        batch_inputs,
        sampling_params=SamplingParams(
            max_tokens=gen_max_tokens,
            temperature=0.0,
            top_p=1.0,
        ),
    )

    questions = [parse_generated_text(output.outputs[0].text) for output in outputs]
    processed_lines = []

    # Create new output entries with only the question
    for qa_idx, line_idx in enumerate(valid_indices):
        q = questions[qa_idx]

        if "error" in q:
            print(f"Warning: {q['error']} in line {line_idx}, skipping item")
            continue

        # Create a new dictionary with only the question
        processed_lines.append(
            {
                "question": q["question"],
                "answer": batch_lines[line_idx].get("improved_code", ""),
            }
        )

    return processed_lines


def main():
    """Main function to process JSONL and generate questions using vLLM with chat template."""
    args = parse_args()

    # Initialize tokenizer and vLLM
    tokenizer = load_tokenizer(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size)

    # Get prompt template
    prompt_template = get_prompt_template(args.lang)

    # Read input JSONL
    lines = read_jsonl(args.input_jsonl)

    # Process in batches
    for start_idx in range(0, len(lines), args.batch_size):
        batch_lines = lines[start_idx : start_idx + args.batch_size]
        processed_lines = process_batch(batch_lines, tokenizer, llm, prompt_template, args.gen_max_tokens)
        if processed_lines:
            save_to_jsonl(args.output_jsonl, processed_lines)


if __name__ == "__main__":
    main()
