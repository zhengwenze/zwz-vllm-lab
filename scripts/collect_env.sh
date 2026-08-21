#!/usr/bin/env bash
set -euo pipefail

echo "== Repository =="
git rev-parse HEAD

echo
echo "== Operating system =="
if [[ -r /etc/os-release ]]; then
  grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
fi
uname -r

echo
echo "== Python and accelerator packages =="
python - <<'PY'
import platform

import flash_attn
import torch
import transformers
import triton

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"triton={triton.__version__}")
print(f"transformers={transformers.__version__}")
print(f"flash_attn={flash_attn.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cxx11_abi={torch.compiled_with_cxx11_abi()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"gpu={props.name}")
    print(f"compute_capability={props.major}.{props.minor}")
PY

echo
echo "== NVIDIA driver =="
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
  --format=csv,noheader
