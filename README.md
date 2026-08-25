# ESPIRiT

PyTorch-based ESPIRiT coil sensitivity calibration for MRI.

Single codebase that runs on **CPU**, **CUDA GPU**, and **Apple Silicon (MPS)** — no separate code paths needed.

## Notice

This package contains a PyTorch translation of the ESPIRiT implementation from the BART (Berkeley Advanced Reconstruction Toolbox), © 2013–2026 The Regents of the University of California and BART Developer Team. BART is licensed under the BSD 3-Clause License. See https://codeberg.org/mrirecon/bart.

## Usage

### 1) CLI
```bash
uvx espirit kspace.npy
```

### 2) Python

```bash
uv add espirit
```

```python
import numpy as np
import torch
from espirit import espirit

# NumPy
kspace_np = np.load("kspace.npy")
csm_np = espirit(kspace_np)

# PyTorch 
kspace_pt = torch.randn(8, 24, 24, 24, dtype=torch.complex64)
csm_pt = espirit(kspace_pt)
```

### Options
```python
csm = espirit(
    kspace,             # (n_coils, *spatial_dims)
    calib_size=24,      # calibration region size
    kernel_size=6,      # sliding-window kernel size
    threshold=0.001,     # singular-value threshold
    mask_threshold=0.8, # eigenvalue mask threshold
    normalize=True,     # RSS=1 normalization
    rotphase=True,      # remove phase ambiguity
    device=None,        # cuda, mps, or cpu (auto-detect when None)
    output_device=None, # final CSM device; use "cpu" to reduce 3D GPU memory
    verbose_memory=False,
)
```

For large 3D datasets, `output_device="cpu"` moves completed sensitivity-map
slices to CPU immediately. This reduces peak GPU memory while keeping the
calibration and eigenmap calculations on the selected compute device. Tensor
inputs return a tensor on `output_device`; NumPy inputs always return a NumPy
array.

## Device support

| Device | Backend | Notes |
|--------|---------|-------|
| `cpu`  | NumPy/MKL | Always available |
| `cuda` | NVIDIA GPU | Requires CUDA toolkit |
| `mps`  | Apple Metal | macOS with Apple Silicon |

The same code runs on all devices — PyTorch handles dispatch automatically.
