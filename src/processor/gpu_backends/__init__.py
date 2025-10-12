from importlib import import_module
from types import ModuleType


_BACKEND_MODULES = {
    "vllm": "src.processor.gpu_backends.vllm_backend",
    "tensorrt-llm": "src.processor.gpu_backends.tensorrt_backend",
}


def load_backend(name: str) -> ModuleType:
    if name not in _BACKEND_MODULES:
        raise ValueError(f"Unsupported GPU backend: {name}")

    module_path = _BACKEND_MODULES[name]
    try:
        return import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(f"GPU backend '{name}' is not available: {exc}") from exc
