# ESPIRiT

PyTorch-based ESPIRiT coil sensitivity calibration for MRI.

Single codebase that runs on **CPU**, **CUDA GPU**, and **Apple Silicon (MPS)** — no separate code paths needed.

## Notice

This package contains a PyTorch translation of the ESPIRiT implementation from the BART (Berkeley Advanced Reconstruction Toolbox), © 2013–2026 The Regents of the University of California and BART Developer Team. BART is licensed under the BSD 3-Clause License. See https://codeberg.org/mrirecon/bart.

## Installation

```bash
uv add espirit
```

## Usage

### Command Line
```bash
uvx espirit kspace.npy
```

### Python API
```python
import numpy as np
import torch
from espirit import espirit

# NumPy
kspace_np = np.load("kspace.npy")
csm_np = espirit(kspace_np)

# PyTorch 
kspace_torch = torch.randn(8, 24, 24, 24, dtype=torch.complex64)
csm_torch = espirit(kspace_torch)
```

### Advanced
All arguments for the `espirit()` function:
```python
csm = espirit(
    kspace,             # (n_coils, *spatial_dims)
    calib_size=24,      # size of calibration region
    kernel_size=6,      # sliding-window kernel size
    threshold=0.01,     # relative singular-value threshold
    mask_threshold=0.8, # eigenvalue mask threshold
    normalize=True,     # RSS=1 normalization
    rotphase=True,      # remove global phase ambiguity
    device=None         # auto-detect (cuda, mps, cpu)
)
```

## Device support

| Device | Backend | Notes |
|--------|---------|-------|
| `cpu`  | NumPy/MKL | Always available |
| `cuda` | NVIDIA GPU | Requires CUDA toolkit |
| `mps`  | Apple Metal | macOS with Apple Silicon |

The same code runs on all devices — PyTorch handles dispatch automatically.
