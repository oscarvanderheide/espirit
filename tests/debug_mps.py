"""Debug MPS vs CPU differences."""

import torch
from espirit import espirit
from conftest import make_synthetic_kspace_2d

torch.manual_seed(42)
kspace = make_synthetic_kspace_2d(n_coils=4, ny=64, nx=64)

# CPU
csm_cpu = espirit(kspace, calib_size=12, kernel_size=6, device="cpu", orthiter=False)
print("CPU CSM stats:")
print(f"  max abs: {csm_cpu.abs().max():.6f}")
print(f"  nonzero: {(csm_cpu.abs() > 1e-6).sum()}/{csm_cpu.numel()}")

# MPS
kspace_mps = kspace.to("mps")
csm_mps = espirit(
    kspace_mps, calib_size=12, kernel_size=6, device="mps", orthiter=False
).cpu()
print("MPS CSM stats:")
print(f"  max abs: {csm_mps.abs().max():.6f}")
print(f"  nonzero: {(csm_mps.abs() > 1e-6).sum()}/{csm_mps.numel()}")

# Where both are nonzero
mask_both = (csm_cpu.abs().sum(0) > 1e-6) & (csm_mps.abs().sum(0) > 1e-6)
print(f"\nBoth nonzero: {mask_both.sum()} voxels")
if mask_both.sum() > 0:
    diff = (csm_cpu[:, mask_both].abs() - csm_mps[:, mask_both].abs()).abs()
    print(f"  Max diff in shared region: {diff.max():.6f}")
    print(f"  Mean diff in shared region: {diff.mean():.6f}")

# Without mask (threshold=0)
print("\nWith threshold=0.0:")
csm_cpu2 = espirit(
    kspace, calib_size=12, kernel_size=6, device="cpu", orthiter=False, threshold=0.0
)
csm_mps2 = espirit(
    kspace_mps,
    calib_size=12,
    kernel_size=6,
    device="mps",
    orthiter=False,
    threshold=0.0,
).cpu()
print(f"  CPU nonzero: {(csm_cpu2.abs() > 1e-6).sum()}/{csm_cpu2.numel()}")
print(f"  MPS nonzero: {(csm_mps2.abs() > 1e-6).sum()}/{csm_mps2.numel()}")
diff2 = (csm_cpu2.abs() - csm_mps2.abs()).abs()
print(f"  Max diff: {diff2.max():.6f}")
print(f"  Mean diff: {diff2.mean():.6f}")
