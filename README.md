<div align="center">

# Swallow-Code-v2

### High-Quality Open Pre-training Code Corpus (Python, C, C++, CUDA)

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Swallow--Code--v2-yellow)](https://huggingface.co/datasets/Swallow-Code-v2)

</div>

## Overview

**Swallow-Code-v2** is a data processing pipeline that transforms raw Python, C, C++, and CUDA source code from "The Stack v2" dataset into a high-quality, cleaned corpus suitable for pre-training large language models. The pipeline implements a multi-stage quality improvement workflow that combines automated formatting, LLM-based error correction, and quality scoring.

### Key Features

- 🌐 **Multi-language Support**: Python, C, C++, and CUDA pipelines with language-specific tooling
- 🔧 **Automated Quality Improvement**: LLM-powered error fixing and code enhancement
- 📊 **Quality Scoring**: Comprehensive evaluation across readability, modularity, clarity, and reusability
- ⚡ **HPC-Optimized**: Designed for large-scale processing on high-performance computing clusters
- 🎯 **Educational Focus**: Produces well-documented, maintainable code examples

## Supported Languages

| Language | Formatter | Linter / Check | Status |
|----------|-----------|----------------|--------|
| Python | Ruff | Ruff | ✅ Full Support |
| C | clang-format | gcc -fsyntax-only | ✅ Full Support |
| C++ | clang-format | g++ -fsyntax-only | ✅ Full Support |
| CUDA C | clang-format | nvcc -c | ✅ Full Support |
| Go | gofmt | go tool compile | ✅ Full Support |
| Rust | rustfmt | rustc --emit metadata | ✅ Full Support |
| JavaScript | prettier | node --check | ✅ Full Support |
| TypeScript | prettier | tsc --noEmit | ✅ Full Support |

## Installation

### Prerequisites
- Python 3.12+
- HPC environment with PBS/Slurm job scheduling
- GPU access for LLM-based stages

### Setup
```bash
# Clone the repository
git clone git@github.com:rioyokotalab/swallow-code-v2.git
cd swallow-code-v2


# vLLM bakcend
# Currently supports: Python stage 2, 4, 5, 8
uv venv .venv
uv pip install transformers vllm

# TensorRT-LLM backend
# Currently supports: Python stage 2, 4
module load cuda/12.8/12.8.1
module load gcc

python -m venv .trt
source .trt/bin/activate
uv pip install -r requirements.txt
uv pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install tensorrt_llm==1.1.0rc5
uv pip install mpi4py openmpi
```

### Dependencies

- `datasets>=3.6.0` - Dataset processing
- `pyarrow>=20.0.0` - Efficient data I/O
- `ruff>=0.11.10` - Python formatting and linting
- `torch>=2.6.0` - PyTorch framework
- `vllm==0.9.1` - Efficient LLM inference

## Usage

### GPU Backend Selection

GPU stages now require an explicit backend selection. Use the `--gpu-backend` flag to choose between the built-in implementations:

- `vllm` (default): implemented in `src/processor/gpu_backends/vllm_backend.py`
- `tensorrt-llm`: implemented in `src/processor/gpu_backends/tensorrt_backend.py`

If the flag is omitted the pipeline falls back to `vllm`. When running TensorRT-LLM jobs, pass the flag explicitly (also reflected in `scripts/python/stage3.sh` and `scripts/python/stage4.sh`):

```bash
python src/pipeline.py gpu --process-stage 3 --gpu-backend tensorrt-llm ...
```

Both backends expose a compatible async client interface, so no additional changes are required to the pipeline besides selecting the flag.

### Running the Complete Pipeline

Submit the stage scripts in sequence as needed. Example commands are available under `scripts/python/`.


### Monitoring Progress

Check job outputs in the `outputs/` directory:

```bash
# View processing logs
tail -f outputs/pipeline/stage1_python/*.OU

# Check intermediate results
ls outputs/dataset/
```

## Development

### Testing

Run the Python test suite with `pytest` to validate the CPU helpers, GPU stubs, prompt loader, and utility functions:

```bash
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Swallow-Code-v2 in your research, please cite:

```bibtex
@dataset{swallow_code_v2_2024,
  title={Swallow-Code-v2: High-Quality Open Pre-training Code Corpus},
  author={[...]},
  year={2025},
  url={https://huggingface.co/datasets/Swallow-Code-v2}
}
```

## Acknowledgments

- Built upon [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) dataset
- Uses [vLLM](https://github.com/vllm-project/vllm) for efficient inference
- Powered by [Qwen](https://github.com/QwenLM/Qwen) language models

---

<div align="center">
Made for the open-source community
</div>
