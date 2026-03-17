# ESPIRiT

PyTorch-based ESPIRiT coil sensitivity calibration for MRI.

Single codebase that runs on **CPU**, **CUDA GPU**, and **Apple Silicon (MPS)** — no separate code paths needed.

## Installation

```bash
pip install -e .
```

## Usage

```python
import torch
from espirit import espirit, select_device

# kspace: (n_coils, nz, ny, nx) complex tensor or NumPy array
device = select_device()  # auto-detects best available: cuda > mps > cpu
csm = espirit(kspace, device=device)
# csm shape: (n_coils, nz, ny, nx), complex64
```

## Device support

| Device | Backend | Notes |
|--------|---------|-------|
| `cpu`  | NumPy/MKL | Always available |
| `cuda` | NVIDIA GPU | Requires CUDA toolkit |
| `mps`  | Apple Metal | macOS with Apple Silicon |

The same code runs on all devices — PyTorch handles dispatch automatically.
