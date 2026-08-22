# Reproducible WSL2 environment

This is the environment used for the first `zwz-vllm-lab` baseline. It is kept
separate from the Windows Conda environments and from the earlier Day 15 vLLM
service.

## Hardware and host

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 |
| GPU memory | 8188 MiB reported by `nvidia-smi` |
| Compute capability | 8.9 |
| NVIDIA driver | 560.94 |
| WSL distribution | Ubuntu 26.04 LTS, WSL2 |
| WSL kernel | 6.18.33.2-microsoft-standard-WSL2 |

## Python runtime

| Package | Version |
| --- | --- |
| Python | 3.12.14 |
| PyTorch | 2.7.1+cu126 |
| CUDA runtime bundled with PyTorch | 12.6 |
| Triton | 3.3.1 |
| Transformers | 5.15.1 |
| FlashAttention | 2.8.3 |
| C++11 ABI | enabled |

The virtual environment lives at:

```text
/home/zwz2025/projects/zwz-vllm-lab/.venv
```

## Reproduce the environment

Run these commands inside the WSL2 `Ubuntu` distribution from the repository
root. `uv` is used only as a fast package installer; a standard Python virtual
environment is still created.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install torch==2.7.1 triton==3.3.1 \
  --index https://download.pytorch.org/whl/cu126
uv pip install transformers==5.15.1 xxhash
uv pip install \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
uv pip install -e . --no-deps
```

The exact FlashAttention wheel matters: it must match Python 3.12, PyTorch 2.7,
CUDA 12.x, and the enabled C++11 ABI. The newest PyTorch wheel available from
the default package index was not used because its CUDA runtime was newer than
the installed NVIDIA driver supports.

Download the model outside the Git repository:

```bash
huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir /home/zwz2025/huggingface/Qwen3-0.6B
```

Activate the environment in a new shell and record it with:

```bash
cd /home/zwz2025/projects/zwz-vllm-lab
source .venv/bin/activate
./scripts/collect_env.sh
```

## Smoke test

`example.py` currently expects the model at `~/huggingface/Qwen3-0.6B/`.

```bash
python example.py
```

The smoke test was completed successfully on 2026-08-21. It loaded the real
Qwen3-0.6B weights and executed the FlashAttention/Triton kernels on the RTX
4060. Output quality is not treated as a correctness metric for this small
model; successful loading and token generation are the acceptance criteria.

