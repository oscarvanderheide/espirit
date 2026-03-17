# ESPIRiT

PyTorch-based ESPIRiT coil sensitivity calibration for MRI.

Single codebase that runs on **CPU**, **CUDA GPU**, and **Apple Silicon (MPS)** — no separate code paths needed.

## Notice

This package contains a PyTorch translation of the ESPIRiT implementation from the BART (Berkeley Advanced Reconstruction Toolbox), © 2013–2026 The Regents of the University of California and BART Developer Team. BART is licensed under the BSD 3-Clause License. See https://codeberg.org/mrirecon/bart.

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
