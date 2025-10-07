<div align="center">

# Swallow-Code-v2

### High-Quality Open Pre-training Code Corpus with 13 Programming Languages

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Swallow--Code--v2-yellow)](https://huggingface.co/datasets/Swallow-Code-v2)

</div>

## Overview

**Swallow-Code-v2** is a comprehensive data processing pipeline that transforms raw source code from "The Stack v2" dataset into a high-quality, cleaned code corpus suitable for pre-training large language models. The pipeline implements a sophisticated 4-stage quality improvement workflow that combines automated formatting, LLM-based error correction, and quality scoring across 13 programming languages.

### Key Features

- 🌐 **Multi-language Support**: 13 programming languages with language-specific tooling
- 🔧 **Automated Quality Improvement**: LLM-powered error fixing and code enhancement  
- 📊 **Quality Scoring**: Comprehensive evaluation across readability, modularity, clarity, and reusability
- ⚡ **HPC-Optimized**: Designed for large-scale processing on high-performance computing clusters
- 🎯 **Educational Focus**: Produces well-documented, maintainable code examples

## Supported Languages

| Language | Formatter | Linter | Status |
|----------|-----------|--------|--------|
| Python | Ruff | Ruff | ✅ Full Support |
| Rust | rustfmt | clippy | ✅ Full Support |
| Java | Google Java Format | Checkstyle | ✅ Full Support |
| C | - | - | 🔧 Basic Support |
| C++ | - | - | 🔧 Basic Support |
| Go | - | - | 🔧 Basic Support |
| JavaScript | - | - | 🔧 Basic Support |
| TypeScript | - | - | 🔧 Basic Support |
| PHP | - | - | 🔧 Basic Support |
| Swift | - | - | 🔧 Basic Support |
| SQL | - | - | 🔧 Basic Support |
| Shell | - | - | 🔧 Basic Support |

## Pipeline Architecture

### Stage 1: Format
**Auto-formatting with language-specific tools**
- Input: Raw JSONL files from The Stack v2
- Process: CPU-based syntax checking and code formatting
- Output: Files with `text_formatted` field and initial `lint_report`
- Infrastructure: CPU nodes, 32 cores, 72-hour limit

### Stage 2: Auto-fix
**LLM-based automatic error correction**
- Input: Formatted code with identified errors
- Process: Uses Qwen3-30B-A3B model to fix syntax and linting errors
- Features: Code block extraction, re-validation, success tracking
- Output: `train_XXXX_without_errors.jsonl` files
- Infrastructure: Single GPU, 72-hour limit

### Stage 3: Long Context Sampling
**Length-based code separation**
- Input: Error-free code samples
- Process: Text length calculation (20,480 token threshold)
- Purpose: Manages context length for downstream processing
- Infrastructure: CPU-based, 72-hour limit

### Stage 4: Quality Scoring
**Comprehensive code quality evaluation**
- Input: Length-filtered code samples
- Evaluation Criteria:
  - **Readability**: Comments, docstrings, formatting, naming
  - **Modularity**: Function separation, component design
  - **Clarity**: Minimal repetition, clear intentions
  - **Reusability**: Error-free, complete, minimal hard-coding
- Output: Quality-scored code with detailed evaluation
- Infrastructure: 8 GPUs, 72-hour limit

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
uv sync

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

### Running the Complete Pipeline

1. **Stage 1 - Format**:

```bash
# Submit formatting job
qsub scripts/abci/pipeline/python/stage1_format.sh
```

2. **Stage 2 - Auto-fix**:

```bash
# Submit error fixing job
qsub scripts/abci/pipeline/python/stage2_auto_fix.sh
```

3. **Stage 3 - Context Sampling**:

```bash
# Submit length filtering job
qsub scripts/abci/pipeline/python/stage3.sh
```

4. **Stage 4 - Quality Scoring**:

```bash
# Submit quality evaluation job
qsub scripts/abci/pipeline/python/stage4_score.sh
```

### Configuration

Edit the shell scripts to customize:

- Input/output paths
- Batch sizes
- Model configurations
- Resource allocation

### Monitoring Progress

Check job outputs in the `outputs/` directory:

```bash
# View processing logs
tail -f outputs/pipeline/stage1_python/*.OU

# Check intermediate results
ls outputs/dataset/
```

## Data Format

### Output Format (Swallow-Code-v2)

dummy example

```json
{
  "content": "def hello():\n    print('Hello, World!')",
  "text_formatted": "def hello():\n    \"\"\"Print a greeting message.\"\"\"\n    print('Hello, World!')",
  "lint_report": {"errors": 0, "warnings": 0},
  "quality_score": 8,
  "quality_reasoning": "Well-structured function with clear purpose...",
  "repo_name": "example/repo",
  "path": "src/hello.py",
  "language": "Python"
}
```

## Quality Metrics

The pipeline evaluates code across four dimensions:

- **Readability** (25%): Documentation, formatting, naming conventions
- **Modularity** (25%): Function separation, component design
- **Clarity** (25%): Code simplicity, intention clarity
- **Reusability** (25%): Completeness, flexibility, error-free execution

Final scores range from 1-10, with detailed reasoning provided for each evaluation.

## Performance

### Throughput

- **Stage 1**: ~xx files/day (CPU-based formatting)
- **Stage 2**: ~xx files/day (LLM-based fixing)
- **Stage 4**: ~xx files/day (Quality scoring)

### Quality Improvements

- **Error Reduction**: xx of syntax errors automatically fixed
- **Documentation**: xx% increase in code comments and docstrings
- **Consistency**: xx% adherence to language formatting standards

### Adding New Languages

1. Create a language module in `src/languages/`
2. Implement formatting and linting tools
3. Add pipeline scripts in `scripts/abci/pipeline/`
4. Update configuration files

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
