"""Debug MPS pipeline step by step."""

import torch
from espirit.espirit import (
    _extract_calibration_region,
    _build_calibration_matrix,
    _compute_kernel_subspace,
    _transform_kernels_to_image_domain,
    _compute_image_domain_covariance,
    _compute_eigenmaps_batched,
)
from conftest import make_synthetic_kspace_2d

torch.manual_seed(42)
kspace = make_synthetic_kspace_2d(n_coils=4, ny=64, nx=64)

calib_size = (12, 12)
kernel_size = (6, 6)
n_coils = 4

for dev_name in ["cpu", "mps"]:
    print(f"\n=== {dev_name.upper()} ===")
    dev = torch.device(dev_name)
    ks = kspace.to(dev)

    calib = _extract_calibration_region(ks, calib_size)
    print(f"calib: {calib.shape}, max abs: {calib.abs().max():.4f}")

    cal_mat = _build_calibration_matrix(calib, kernel_size)
    print(f"cal_mat: {cal_mat.shape}, max abs: {cal_mat.abs().max():.4f}")

    kernels, svals = _compute_kernel_subspace(cal_mat)
    print(f"kernels: {kernels.shape}, svals: {svals[:5]}")
    print(f"n_keep: {len(svals)}")

    img_kernels = _transform_kernels_to_image_domain(kernels, kernel_size, n_coils)
    print(f"img_kernels: {img_kernels.shape}, max abs: {img_kernels.abs().max():.4f}")

    cov = _compute_image_domain_covariance(img_kernels, kernel_size)
    print(f"cov: {cov.shape}, max abs: {cov.abs().max():.4f}")

    # Check if cov looks reasonable
    cov_diag = torch.diagonal(cov, dim1=-2, dim2=-1)
    print(
        f"cov diagonal max: {cov_diag.abs().max():.4f}, mean: {cov_diag.abs().mean():.4f}"
    )

    csm, eigenvalues = _compute_eigenmaps_batched(cov, orthiter=False, num_orthiter=30)
    print(f"csm shape: {csm.shape}, max abs: {csm.abs().max():.4f}")
    print(
        f"eigenvalues shape: {eigenvalues.shape}, max: {eigenvalues.max():.4f}, mean: {eigenvalues.mean():.4f}"
    )
    print(f"eigenvalues > 0.95: {(eigenvalues > 0.95).sum()}/{eigenvalues.numel()}")
