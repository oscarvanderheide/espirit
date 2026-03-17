"""Debug: compare image kernels and covariance on CPU vs MPS."""

import torch
from espirit.espirit import (
    _extract_calibration_region,
    _build_calibration_matrix,
    _compute_kernel_subspace,
    _transform_kernels_to_image_domain,
    _compute_image_domain_covariance,
)
from conftest import make_synthetic_kspace_2d

torch.manual_seed(42)
kspace = make_synthetic_kspace_2d(n_coils=4, ny=64, nx=64)

calib_size = (12, 12)
kernel_size = (6, 6)
n_coils = 4

# CPU pipeline
calib_cpu = _extract_calibration_region(kspace, calib_size)
cal_mat_cpu = _build_calibration_matrix(calib_cpu, kernel_size)
kernels_cpu, svals_cpu = _compute_kernel_subspace(cal_mat_cpu)
img_kern_cpu = _transform_kernels_to_image_domain(kernels_cpu, kernel_size, n_coils)
cov_cpu = _compute_image_domain_covariance(img_kern_cpu, kernel_size)

# MPS pipeline
ks_mps = kspace.to("mps")
calib_mps = _extract_calibration_region(ks_mps, calib_size)
cal_mat_mps = _build_calibration_matrix(calib_mps, kernel_size)
kernels_mps, svals_mps = _compute_kernel_subspace(cal_mat_mps)
img_kern_mps = _transform_kernels_to_image_domain(kernels_mps, kernel_size, n_coils)
cov_mps = _compute_image_domain_covariance(img_kern_mps, kernel_size)

# Compare kernels
diff_kern = (img_kern_cpu - img_kern_mps.cpu()).abs()
print(f"img_kernels diff: max={diff_kern.max():.6e}, mean={diff_kern.mean():.6e}")

# Compare covariance
diff_cov = (cov_cpu - cov_mps.cpu()).abs()
print(f"covariance diff: max={diff_cov.max():.6e}, mean={diff_cov.mean():.6e}")

# Check normalization factor
print(f"\nkernel_elements: {6 * 6} = 36")
print(f"img_elements: {12 * 12} = 144")
print(f"normalization: {36 / 144**2}")

# Manually compute covariance at position (6,6) and compare
pos = (6, 6)
H_cpu = img_kern_cpu[:, :, pos[0], pos[1]]  # (n_kernels, n_coils)
H_mps = img_kern_mps[:, :, pos[0], pos[1]].cpu()  # (n_kernels, n_coils)
print(f"\nAt position {pos}:")
print(f"  H_cpu shape: {H_cpu.shape}, max: {H_cpu.abs().max():.6e}")
print(f"  H_mps shape: {H_mps.shape}, max: {H_mps.abs().max():.6e}")
print(f"  H diff: {(H_cpu - H_mps).abs().max():.6e}")

manual_cov_cpu = H_cpu.conj().T @ H_cpu / (36 / 144**2)
manual_cov_mps = H_mps.conj().T @ H_mps / (36 / 144**2)
print(f"  manual cov_cpu diag: {manual_cov_cpu.diag().abs()}")
print(f"  manual cov_mps diag: {manual_cov_mps.diag().abs()}")
print(f"  cov_cpu at {pos}: {cov_cpu[pos[0], pos[1]].diag().abs()}")
print(f"  cov_mps at {pos}: {cov_mps[pos[0], pos[1]].cpu().diag().abs()}")
