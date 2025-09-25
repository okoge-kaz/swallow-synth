import argparse
import json
import os
from typing import Dict, List, Union
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, PreTrainedTokenizer

# Define the prompt templates for translating both question and code as message structures for chat templates

# For translating questions (English to Japanese)
TRANSLATE_QUESTION_MESSAGES: List[Dict[str, str]] = [
    {
        "role": "system",
        "content": "You are a helpful assistant tasked with translating detailed English problem statements (Questions) into natural and accurate Japanese, preserving all technical details, constraints, and requirements.",
    },
    {
        "role": "user",
        "content": """Here is a few-shot example:

English Question: Implement a RESTful API endpoint in a Flask-RESTful application that assigns the next available subtask to an idle machine in a QA automation system, ensuring thread-safe database operations and respecting task preconditions and machine labels. The endpoint must:

1. Accept optional query parameters: `task` (filter by major task name), `tracknumber` (filter by track number), and `status` (filter by status).
2. Use database row-level locking (via `with_for_update()`) on the Machine table to prevent race conditions during assignment.
3. Find the next valid (subtask, machine) pair using the following priority order:
 - First, assign any "report" subtask to an available Windows machine (if one exists).
 - Then, assign subtasks with no preconditions (precondition = "no").
 - Then, assign subtasks whose preconditions are satisfied (i.e., the precondition subtask has status 2 and result ≠ "failed").
4. Match subtasks to machines based on label requirements:
 - If a subtask has a `label` property, assign it only to a machine with that label.
 - If no label is specified, default to "windows".
5. Update the database to reflect the assignment:
 - Set the subtask status to 1 ("doing").
 - Assign the machine's hostname to the subtask.
 - Set the machine status to 1 ("busy").
 - If the major task was in "todo" (status 0), update it to "doing" (status 1).
6. Return a JSON response with:
 - `task_name`: The major task's track number.
 - `subtask_type`: The subtask name.
 - `machine`: The machine's IP address.
 - If no assignment is possible, return `None` values for all fields.
7. Handle all database errors gracefully with appropriate HTTP 500 responses and rollback transactions.
8. Ensure the solution is thread-safe and handles edge cases such as:
 - No idle machines.
 - No pending subtasks.
 - Precondition subtasks not found or not completed.
 - Database connection failures.
9. Use a `Session` from a configured database factory (`database.Session`) and ensure the session is properly closed in a `finally` block.
10. The solution must be efficient with O(n) time complexity in the worst case, where n is the number of subtasks and machines, and use minimal memory (no unnecessary data duplication).

Japanese Question: Flask-RESTfulアプリケーションでRESTful APIエンドポイントを実装し、QA自動化システムでアイドル状態のマシンに次の利用可能なサブタスクを割り当ててください。スレッドセーフなデータベース操作を確保し、タスクの前提条件とマシンのラベルを尊重します。このエンドポイントは以下の要件を満たす必要があります：

1. オプションのクエリパラメータを受け取る：`task`（メジャータスク名によるフィルタ）、`tracknumber`（トラック番号によるフィルタ）、`status`（ステータスによるフィルタ）。
2. 割り当て中のレースコンディションを防ぐために、Machineテーブルでデータベースの行レベルロック（`with_for_update()`経由）を使用。
3. 次の有効な（サブタスク、マシン）のペアを以下の優先順位で探す：
 - まず、"report"サブタスクを利用可能なWindowsマシン（存在する場合）に割り当てる。
 - 次に、前提条件のないサブタスク（precondition = "no"）を割り当てる。
 - 次に、前提条件が満たされたサブタスク（前提条件サブタスクがステータス2でresult ≠ "failed"）を割り当てる。
4. ラベル要件に基づいてサブタスクをマシンにマッチさせる：
 - サブタスクに`label`プロパティがある場合、そのラベルを持つマシンのみに割り当てる。
 - ラベルが指定されていない場合、デフォルトで"windows"とする。
5. 割り当てを反映してデータベースを更新：
 - サブタスクステータスを1（"doing"）に設定。
 - サブタスクにマシンのホスト名を割り当てる。
 - マシンステータスを1（"busy"）に設定。
 - メジャータスクが"todo"（ステータス0）だった場合、"doing"（ステータス1）に更新。
6. JSONレスポンスを返す：
 - `task_name`：メジャータスクのトラック番号。
 - `subtask_type`：サブタスク名。
 - `machine`：マシンのIPアドレス。
 - 割り当てが不可能な場合、全フィールドに`None`値を返す。
7. すべてのデータベースエラーを適切に処理し、HTTP 500レスポンスとトランザクションロールバックを行う。
8. ソリューションがスレッドセーフであり、以下のエッジケースを扱うことを確保：
 - アイドルマシンが存在しない。
 - 保留中のサブタスクが存在しない。
 - 前提条件サブタスクが見つからないまたは完了していない。
 - データベース接続失敗。
9. 設定されたデータベースファクトリ（`database.Session`）から`Session`を使用し、`finally`ブロックでセッションを適切に閉じる。
10. ソリューションは最悪ケースでO(n)時間複雑度（nはサブタスクとマシンの数）で効率的であり、最小限のメモリを使用（不必要なデータ複製なし）。

Now, given the following English Question, translate it into Japanese. Make the translation natural, accurate, and maintain all key conditions, behaviors, constraints, input/output formats, edge cases, and complexity requirements.

English Question:
{QUESTION_CONTENT}

Provide the Japanese Question right after "Japanese Question: ".""",
    },
]

# For translating code docstrings and comments (English to Japanese)
TRANSLATE_CODE_MESSAGES: List[Dict[str, str]] = [
    {
        "role": "system",
        "content": "You are a helpful assistant tasked with translating docstrings and comments in Python code from English to Japanese, while keeping the code logic, structure, and type annotations intact.",
    },
    {
        "role": "user",
        "content": """Here is a few-shot example:

Original Code:
def assign_subtask(task: Optional[str] = None, tracknumber: Optional[int] = None, status: Optional[int] = None) -> Dict[str, Union[str, int, None]]:
    \"\"\"Assign the next available subtask to an idle machine.

    This function implements the logic for assigning subtasks in a QA automation system.
    It ensures thread-safety using database locks and respects preconditions and labels.

    Args:
        task (Optional[str]): Filter by major task name.
        tracknumber (Optional[int]): Filter by track number.
        status (Optional[int]): Filter by status.

    Returns:
        Dict[str, Union[str, int, None]]: JSON response with task_name, subtask_type, machine or None values.

    Raises:
        DatabaseError: If a database connection or query fails.
    \"\"\"
    session = database.Session()
    try:
        # Lock idle machines to prevent race conditions
        machines = session.query(Machine).filter(Machine.status == 0).with_for_update().all()
        if not machines:
            return {"task_name": None, "subtask_type": None, "machine": None}  # No idle machines available

        # Query pending subtasks with filters
        subtasks = session.query(Subtask).filter(Subtask.status == 0)
        if task:
            subtasks = subtasks.filter(Subtask.task == task)
        # Additional filtering logic...
        return {"task_name": "example", "subtask_type": "report", "machine": "192.168.1.1"}
    except Exception as e:
        session.rollback()
        raise DatabaseError("Database error occurred") from e
    finally:
        session.close()

Translated Code:
def assign_subtask(task: Optional[str] = None, tracknumber: Optional[int] = None, status: Optional[int] = None) -> Dict[str, Union[str, int, None]]:
    \"\"\"アイドル状態のマシンに次の利用可能なサブタスクを割り当てる。

    この関数はQA自動化システムでのサブタスク割り当てロジックを実装します。
    データベースロックを使用してスレッドセーフを確保し、前提条件とラベルを尊重します。

    Args:
        task (Optional[str]): メジャータスク名によるフィルタ。
        tracknumber (Optional[int]): トラック番号によるフィルタ。
        status (Optional[int]): ステータスによるフィルタ。

    Returns:
        Dict[str, Union[str, int, None]]: task_name, subtask_type, machineを含むJSONレスポンス、またはNone値。

    Raises:
        DatabaseError: データベース接続またはクエリが失敗した場合。
    \"\"\"
    session = database.Session()
    try:
        # レースコンディションを防ぐためにアイドルマシンをロック
        machines = session.query(Machine).filter(Machine.status == 0).with_for_update().all()
        if not machines:
            return {"task_name": None, "subtask_type": None, "machine": None}  # 利用可能なアイドルマシンがありません

        # フィルタ付きで保留中のサブタスクをクエリ
        subtasks = session.query(Subtask).filter(Subtask.status == 0)
        if task:
            subtasks = subtasks.filter(Subtask.task == task)
        # 追加のフィルタリングロジック...
        return {"task_name": "example", "subtask_type": "report", "machine": "192.168.1.1"}
    except Exception as e:
        session.rollback()
        raise DatabaseError("データベースエラーが発生しました") from e
    finally:
        session.close()

Now, given the following Python code, translate all docstrings and comments into Japanese. Keep the code structure, logic, type annotations, and any non-text elements unchanged. Only translate the text in docstrings (triple-quoted strings) and comments (lines starting with # or inline comments).

Code:
{CODE_CONTENT}

Provide the translated code right after "Translated Code: ".""",
    },
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with clear defaults and type hints.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing input/output paths and model settings.
    """
    parser = argparse.ArgumentParser(
        description="Translate both questions and code elements (docstrings/comments) from English to Japanese in JSONL using vLLM."
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Path to input JSONL file containing dictionaries with 'question' and 'answer' keys.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="Path to output JSONL file where translated results will be saved.",
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to the vLLM model directory.")
    parser.add_argument(
        "--batch-size", type=int, default=4096, help="Number of JSONL lines to process in each batch (default: 4096)."
    )
    parser.add_argument(
        "--gen-max-tokens",
        type=int,
        default=16384,
        help="Maximum number of tokens for generated output (default: 16384).",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for vLLM model (default: 1)."
    )
    return parser.parse_args()


def load_tokenizer(model_path: str) -> PreTrainedTokenizer:
    """Load and return a tokenizer for the specified model.

    Args:
        model_path (str): Path to the model directory containing tokenizer configuration.

    Returns:
        PreTrainedTokenizer: Loaded tokenizer instance for the specified model.

    Raises:
        Exception: If the tokenizer cannot be loaded from the model path.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def read_jsonl(file_path: str) -> List[Dict[str, Union[str, List]]]:
    """Read and parse JSONL file into a list of dictionaries.

    Args:
        file_path (str): Path to the JSONL file to be read.

    Returns:
        List[Dict[str, Union[str, List]]]: List of dictionaries parsed from the JSONL file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        json.JSONDecodeError: If any line in the JSONL file is invalid JSON.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_to_jsonl(file_path: str, data_list: List[Dict[str, Union[str, List]]]) -> None:
    """Append data to a JSONL file.

    Args:
        file_path (str): Path to the output JSONL file.
        data_list (List[Dict[str, Union[str, List]]]): List of dictionaries to append to the file.

    Raises:
        IOError: If there is an error writing to the file.
    """
    mode = "a" if os.path.exists(file_path) else "w"
    with open(file_path, mode, encoding="utf-8") as f:
        for entry in data_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def parse_generated_text(text: str, sub_mode: str) -> Dict[str, str]:
    """Parse generated text based on sub-mode (question or code).

    Args:
        text (str): The generated text from the model.
        sub_mode (str): The sub-processing mode ('question' or 'code').

    Returns:
        Dict[str, str]: Dictionary containing the parsed 'question' or 'code', or an 'error' key if parsing fails.

    Raises:
        ValueError: If the expected marker is not found or the parsed content is empty.
    """
    print(f"Generated Text:\n{text}\n{'-' * 40}", flush=True)
    result: Dict[str, str] = {}

    try:
        if sub_mode == "question":
            marker = "Japanese Question: "
            start_idx = text.find(marker)
            if start_idx == -1:
                raise ValueError("No 'Japanese Question: ' marker found in generated text")
            question_text = text[start_idx + len(marker) :].strip()
            if not question_text:
                raise ValueError("Parsed question text is empty")
            result["question"] = question_text
        elif sub_mode == "code":
            marker = "Translated Code:"
            start_idx = text.find(marker)
            if start_idx == -1:
                raise ValueError("No 'Translated Code: ' marker found in generated text")
            code_text = text[start_idx + len(marker) :].strip()
            if not code_text:
                raise ValueError("Parsed code text is empty")
            result["code"] = code_text
        else:
            raise ValueError(f"Invalid sub_mode: {sub_mode}")
    except Exception as e:
        result = {"error": f"Failed to parse generated text: {str(e)}"}

    return result


def process_batch(
    batch_lines: List[Dict[str, Union[str, List]]],
    tokenizer: PreTrainedTokenizer,
    llm: LLM,
    gen_max_tokens: int,
) -> List[Dict[str, Union[str, List]]]:
    """Process a batch of JSONL lines and generate translations for both questions and code docstrings/comments.

    Args:
        batch_lines (List[Dict[str, Union[str, List]]]): Batch of input data from JSONL, each containing 'question' and 'answer' keys.
        tokenizer (PreTrainedTokenizer): Tokenizer for formatting prompts with chat templates.
        llm (LLM): vLLM instance for generating translations.
        gen_max_tokens (int): Maximum number of tokens for generated output.

    Returns:
        List[Dict[str, Union[str, List]]]: List of processed dictionaries with translated 'question' and 'answer' fields.

    Raises:
        ValueError: If prompt construction or parsing fails.
    """
    batch_inputs: List[str] = []
    valid_indices: List[int] = []
    sub_modes: List[str] = []  # Track whether each input is for 'question' or 'code'

    question_template = TRANSLATE_QUESTION_MESSAGES
    code_template = TRANSLATE_CODE_MESSAGES

    for local_idx, item in enumerate(batch_lines):
        question_content = item.get("question", "")
        code_content = item.get("answer", "")

        # Process question if available
        if question_content:
            formatted_messages = [msg.copy() for msg in question_template]
            for msg in formatted_messages:
                if msg["role"] == "user":
                    # Use string concatenation to avoid .format() issues with curly braces
                    msg["content"] = msg["content"].replace("{QUESTION_CONTENT}", question_content.strip())
            prompt = tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
            batch_inputs.append(prompt)
            valid_indices.append(local_idx)
            sub_modes.append("question")

        # Process code if available
        if code_content:
            formatted_messages = [msg.copy() for msg in code_template]
            for msg in formatted_messages:
                if msg["role"] == "user":
                    # Use string concatenation to avoid .format() issues with curly braces
                    msg["content"] = msg["content"].replace("{CODE_CONTENT}", code_content.strip())
            prompt = tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
            batch_inputs.append(prompt)
            valid_indices.append(local_idx)
            sub_modes.append("code")

    if not batch_inputs:
        return []

    # Generate outputs using vLLM
    outputs = llm.generate(
        batch_inputs,
        sampling_params=SamplingParams(
            max_tokens=gen_max_tokens,
            temperature=0.0,
            top_p=1.0,
        ),
    )

    results = [parse_generated_text(output.outputs[0].text, sub_modes[i]) for i, output in enumerate(outputs)]
    temp_results: Dict[int, Dict[str, str]] = {}  # Store results per line_idx

    for i, line_idx in enumerate(valid_indices):
        res = results[i]
        if "error" in res:
            print(f"Warning: {res['error']} in line {line_idx} for {sub_modes[i]}, skipping part")
            continue

        if line_idx not in temp_results:
            temp_results[line_idx] = {}

        if sub_modes[i] == "question":
            temp_results[line_idx]["question"] = res["question"]
        elif sub_modes[i] == "code":
            temp_results[line_idx]["answer"] = res["code"]

    # Construct final output
    processed_lines: List[Dict[str, Union[str, List]]] = []
    for line_idx in set(valid_indices):
        if line_idx in temp_results:
            original_item = batch_lines[line_idx]
            output_item = {
                "question": temp_results[line_idx].get("question", original_item.get("question", "")),
                "answer": temp_results[line_idx].get("answer", original_item.get("answer", "")),
            }
            if output_item["question"] or output_item["answer"]:  # Only add if at least one part translated
                processed_lines.append(output_item)

    return processed_lines


def main() -> None:
    """Main function to process JSONL and generate translations for both questions and code docstrings/comments using vLLM.

    Reads input JSONL, processes it in batches, and saves translated results to output JSONL.
    Translates both questions and code docstrings/comments from English to Japanese.

    Raises:
        FileNotFoundError: If the input JSONL file does not exist.
        ValueError: If the model path is invalid or prompt construction fails.
    """
    args = parse_args()

    # Initialize tokenizer and vLLM
    tokenizer = load_tokenizer(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size)

    # Read input JSONL
    lines = read_jsonl(args.input_jsonl)

    # Process in batches
    for start_idx in range(0, len(lines), args.batch_size):
        batch_lines = lines[start_idx : start_idx + args.batch_size]
        processed_lines = process_batch(batch_lines, tokenizer, llm, args.gen_max_tokens)
        if processed_lines:
            save_to_jsonl(args.output_jsonl, processed_lines)


if __name__ == "__main__":
    main()
