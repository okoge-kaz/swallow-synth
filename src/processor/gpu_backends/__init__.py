_BACKEND_MODULES = {"vllm", "tensorrt-llm"}


def load_backend(name: str):
    if name not in _BACKEND_MODULES:
        raise ValueError(f"Unsupported GPU backend: {name}")

    if name == "vllm":
        from src.processor.gpu_backends import vllm_backend

        return vllm_backend
    elif name == "tensorrt-llm":
        from src.processor.gpu_backends import tensorrt_backend

        return tensorrt_backend
    else:
        raise ValueError(f"Unsupported GPU backend: {name}")
