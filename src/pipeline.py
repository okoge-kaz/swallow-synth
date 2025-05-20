import json
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count, Manager, Lock
from multiprocessing.managers import DictProxy
from typing import Callable, Any, Iterator

from src.languages.python import process_item_cpu as python_process_item_cpu, RewritePipeline as PythonRewritePipeline
from src.languages.rust import process_item_cpu as rust_process_item_cpu, RewritePipeline as RustRewritePipeline
from src.languages.java import process_item_cpu as java_process_item_cpu, RewritePipeline as JavaRewritePipeline
from src.languages.c import process_item_cpu as c_process_item_cpu, RewritePipeline as CRewritePipeline
from src.languages.cpp import process_item_cpu as cpp_process_item_cpu, RewritePipeline as CppRewritePipeline


def get_process_item_cpu(lang: str) -> Callable:
    processors = {
        "python": python_process_item_cpu,
        "rust": rust_process_item_cpu,
        "java": java_process_item_cpu,
        "c": c_process_item_cpu,
        "cpp": cpp_process_item_cpu,
    }
    if lang not in processors:
        raise ValueError(f"Unsupported language: {lang}")
    return processors[lang]


def get_rewrite_pipeline(lang: str, model_name: str, batch_size: int, tensor_parallel_size: int = 1):
    pipelines = {
        "python": PythonRewritePipeline,
        "rust": RustRewritePipeline,
        "java": JavaRewritePipeline,
        "c": CRewritePipeline,
        "cpp": CppRewritePipeline,
    }
    if lang not in pipelines:
        raise ValueError(f"Unsupported language: {lang}")
    return pipelines[lang](model_name, tensor_parallel_size)


def get_average_processing_time(history_dict: DictProxy[str, list[float]], lang: str, mode: str) -> float:
    """Calculate average processing time from actual processing time history"""
    key = f"{lang}_{mode}"
    if not history_dict.get(key):
        return 0.0
    return sum(history_dict[key]) / len(history_dict[key])


def update_processing_time(history_dict: DictProxy[str, list[float]], lang: str, mode: str, time_taken: float) -> None:
    """Update processing time history"""
    key = f"{lang}_{mode}"
    if key not in history_dict:
        history_dict[key] = []
    history_dict[key].append(time_taken)
    # Keep only the latest 100 items
    if len(history_dict[key]) > 100:
        history_dict[key] = history_dict[key][-100:]


def write_result(result: dict[str, Any], fout, lock: Any) -> None:
    """Synchronized write operation for results"""
    with lock:
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        fout.flush()


def process_item_with_timing(
    item: dict[str, Any], process_func: Callable, history_dict: DictProxy[str, list[float]], lang: str, fout, lock: Any
) -> None:
    """Process a single item, record its processing time and write the result"""
    start_time = time.time()
    result = process_func(item)
    time_taken = time.time() - start_time
    update_processing_time(history_dict, lang, "cpu", time_taken)
    write_result(result, fout, lock)


def stream_jsonl(file_path: Path, batch_size: int = 1024) -> Iterator[list[dict[str, Any]]]:
    """Stream JSONL file in batches"""
    batch = []
    with file_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:  # return remaining data
            yield batch


def preprocess(input_path: Path, output_path: Path, lang: str, n_workers: int = 16, batch_size: int = 1000) -> None:
    """CPU only processing stage"""
    n_workers = n_workers or cpu_count()
    process_item_cpu = get_process_item_cpu(lang)

    # Create a manager for shared dictionary and lock
    with Manager() as manager:
        history_dict = manager.dict()
        lock = manager.Lock()

        # Process batches
        total_items = 0
        start_time = time.time()

        with Pool(n_workers) as pool, output_path.open("w", encoding="utf-8") as fout:
            for batch in stream_jsonl(input_path, batch_size):
                total_items += len(batch)

                # Past processing time prediction
                avg_time = get_average_processing_time(history_dict, lang, "cpu")
                if avg_time > 0:
                    est_time = avg_time * len(batch) / n_workers
                    print(f"Past average CPU processing time prediction: {est_time:.1f}s ({len(batch)} items)")

                # Process batch
                process_func = lambda item: process_item_with_timing(
                    item, process_item_cpu, history_dict, lang, fout, lock
                )
                pool.map(process_func, batch)

        actual_time = time.time() - start_time
        print(f"CPU processing time: {actual_time:.1f}s (1 item: {actual_time / total_items:.3f}s)")


def rewrite(input_path: Path, output_path: Path, lang: str, model_name: str = "qwen-3", batch_size: int = 32, tensor_parallel_size: int = 1) -> None:
    """GPU only processing stage"""
    pipeline = get_rewrite_pipeline(lang, model_name, batch_size, tensor_parallel_size)

    # Create a manager for shared dictionary
    with Manager() as manager:
        history_dict = manager.dict()
        total_items = 0
        start_time = time.time()

        # Process batches
        with output_path.open("w", encoding="utf-8") as fout:
            for batch in stream_jsonl(input_path, batch_size):
                total_items += len(batch)

                # Past processing time prediction
                avg_time = get_average_processing_time(history_dict, lang, "gpu")
                if avg_time > 0:
                    est_time = avg_time * len(batch)
                    print(f"Past average GPU processing time prediction: {est_time:.1f}s ({len(batch)} items)")

                # Process batch
                batch_start_time = time.time()
                results = pipeline.process_item_gpu(batch)
                batch_time = time.time() - batch_start_time
                update_processing_time(history_dict, lang, "gpu", batch_time / len(batch))

                # Write results
                for result in results:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()

        actual_time = time.time() - start_time
        print(f"GPU processing time: {actual_time:.1f}s (1 item: {actual_time / total_items:.3f}s)")


# === CLI Entrypoint ===

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Code Quality Pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("cpu", help="CPU only processing stage")
    p1.add_argument("--input-jsonl", type=Path, required=True)
    p1.add_argument("--output-jsonl", type=Path, required=True)
    p1.add_argument("--workers", type=int, default=32, help="Number of CPU workers")
    p1.add_argument("--lang", type=str, required=True, help="Programming language (e.g., python, rust, java)")
    p1.add_argument("--batch-size", type=int, default=1000, help="Batch size for CPU processing")

    p2 = sub.add_parser("gpu", help="GPU only processing stage")
    p2.add_argument("--input-jsonl", type=Path, required=True)
    p2.add_argument("--output-jsonl", type=Path, required=True)
    p2.add_argument("--model", type=str, default="qwen-3", help="Local Qwen model identifier for vLLM")
    p2.add_argument("--lang", type=str, required=True, help="Programming language (e.g., python, rust, java)")
    p2.add_argument("--batch-size", type=int, default=32, help="Batch size for GPU processing")
    p2.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use for tensor parallelism")

    args = parser.parse_args()
    if args.cmd == "cpu":
        preprocess(
            input_path=args.input_jsonl,
            output_path=args.output_jsonl,
            n_workers=args.workers,
            lang=args.lang,
            batch_size=args.batch_size,
        )
    elif args.cmd == "gpu":
        rewrite(
            input_path=args.input_jsonl,
            output_path=args.output_jsonl,
            model_name=args.model,
            lang=args.lang,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
        )
