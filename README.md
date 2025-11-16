<div align="center">

# Swallow-Reasoning-v1

### High-Quality Open Pre-training Reasoning Corpus

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## Overview

**Swallow-Reasoning-v1**

### Prerequisites
- Python 3.12+
- HPC environment with PBS/Slurm job scheduling
- GPU access for LLM-based stages

### Setup
```bash
# Clone the repository


# vLLM backend
uv venv .venv
uv pip install transformers vllm
uv pip install vllm[flashinfer]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@dataset{
}
```

## Acknowledgments

- Uses [vLLM](https://github.com/vllm-project/vllm) for efficient inference

---

<div align="center">
Made for the open-source community
</div>
