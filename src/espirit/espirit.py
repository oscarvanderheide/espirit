"""
ESPIRiT coil sensitivity calibration (PyTorch implementation).

Python implementation of ESPIRiT (Uecker et al., MRM 2014).
Runs on CPU, CUDA, and MPS with a single code path.

Pipeline (see espirit()):
  1. Extract calibration region from k-space centre
  2. Build calibration matrix from overlapping k-space patches
  3. Compute signal-space kernels via eigendecomposition of the Gram matrix
  4. Transform kernels to image domain to obtain PSF-like basis functions
  5. Compute per-voxel covariance matrices from image-domain kernels
  6. Sinc-interpolate covariance to full image resolution and extract CSM
     as the dominant eigenvector at each voxel
  7. Mask sensitivity maps based on eigenvalue threshold
  8. Phase-align maps to remove global phase ambiguity
  9. Normalise maps so RSS across coils = 1
"""

from __future__ import annotations

import torch
import numpy as np

from .device import select_device
from .fft import ifft2c, ifft3c


# =============================================================================
# MPS COMPATIBILITY HELPERS
# =============================================================================


def _eigh(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.linalg.eigh with automatic CPU fallback for MPS."""
    try:
        return torch.linalg.eigh(A)
    except NotImplementedError:
        w, v = torch.linalg.eigh(A.cpu())
        return w.to(A.device), v.to(A.device)


def _complex_norm(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """Norm of complex tensor that works on MPS (where linalg.norm may not)."""
    return torch.sqrt(torch.sum(x.real**2 + x.imag**2, dim=dim, keepdim=keepdim))


# =============================================================================
# PUBLIC API
# =============================================================================


def espirit(
    kspace,
    calib_size: int | tuple | None = None,
    kernel_size: int | tuple | None = None,
    threshold: float = 0.001,
    mask_threshold: float = 0.8,
    normalize: bool = True,
    rotphase: bool = True,
    orthiter: bool = True,
    num_orthiter: int = 30,
    soft_threshold: bool = False,
    device: str | torch.device | None = None,
):
    """
    Run the full ESPIRiT calibration pipeline and return coil sensitivity maps.

    Parameters
    ----------
    kspace : ndarray or torch.Tensor, shape (n_coils, *spatial_dims)
        Multi-coil k-space data. Can be 2D or 3D.
    calib_size : int or tuple, optional
        Size of the calibration region. Default: 24.
    kernel_size : int or tuple, optional
        Size of the sliding-window kernel. Default: 6.
    threshold : float
        Relative singular-value threshold for retaining signal-space kernels.
    mask_threshold : float
        Eigenvalue threshold below which sensitivity maps are zeroed out.
    normalize : bool
        Normalise maps so RSS across coils = 1.
    rotphase : bool
        Remove the global phase ambiguity.
    orthiter : bool
        Use power iteration instead of full eigendecomposition.
    num_orthiter : int
        Number of power-iteration steps.
    soft_threshold : bool
        Use a smooth S-curve transition instead of a hard binary mask.
    device : str or torch.device, optional
        Device to run on. Auto-detected when None.

    Returns
    -------
    csm : same type as input (ndarray or Tensor), shape (n_coils, *spatial_dims)
        Coil sensitivity maps.
    """
    # Handle NumPy input — convert, run, convert back
    return_numpy = isinstance(kspace, np.ndarray)
    if return_numpy:
        kspace = torch.from_numpy(kspace)

    if device is None:
        device = select_device()
    device = torch.device(device) if isinstance(device, str) else device

    kspace = kspace.to(device=device, dtype=torch.complex64)

    n_coils = kspace.shape[0]
    n_dims = len(kspace.shape) - 1

    calib_size = _to_size_tuple(calib_size, n_dims, default=24)
    kernel_size = _to_size_tuple(kernel_size, n_dims, default=6)

    # Step 1
    calib_data = _extract_calibration_region(kspace, calib_size)

    # Step 2
    cal_matrix = _build_calibration_matrix(calib_data, kernel_size)

    # Step 3
    kernels, _ = _compute_kernel_subspace(cal_matrix, threshold=threshold)

    # Step 4
    img_kernels = _transform_kernels_to_image_domain(kernels, kernel_size, n_coils)

    # Step 5
    img_cov = _compute_image_domain_covariance(img_kernels, kernel_size)

    # Step 6
    csm, eigenvalues = _interpolate_covariance_and_extract_csm(
        img_cov, kspace.shape[1:], orthiter=orthiter, num_orthiter=num_orthiter
    )

    # Step 7
    csm = _mask_sensitivity_maps(csm, eigenvalues, mask_threshold, soft_threshold)

    # Step 8
    if rotphase:
        rotation_matrix = _build_phase_rotation_matrix(calib_data)
        csm = _apply_phase_rotation(csm, rotation_matrix)

    # Step 9
    if normalize:
        csm = _normalize_sensitivity_maps(csm)

    # Return first set of maps (dominant eigenvector)
    result = csm[0]

    if return_numpy:
        return result.cpu().numpy()
    return result


# =============================================================================
# HELPERS
# =============================================================================


def _to_size_tuple(value, n_dims, default):
    if value is None:
        return tuple([default] * n_dims)
    if isinstance(value, int):
        return tuple([value] * n_dims)
    return tuple(value)


# =============================================================================
# STEPS 1–5: BUILDING THE COVARIANCE MATRIX
# =============================================================================


def _extract_calibration_region(
    kspace: torch.Tensor, calib_size: tuple
) -> torch.Tensor:
    """Extract the central calibration region from k-space."""
    spatial_dims = kspace.shape[1:]
    slices = [slice(None)]  # coil dimension
    for full_size, cal_size_i in zip(spatial_dims, calib_size):
        center = full_size // 2
        start = center - (cal_size_i // 2)
        slices.append(slice(start, start + cal_size_i))
    return kspace[tuple(slices)].contiguous()


def _build_calibration_matrix(
    calib_data: torch.Tensor, kernel_size: tuple
) -> torch.Tensor:
    """
    Build a Casorati-style matrix where each row is a flattened k-space patch.

    Uses unfold to extract patches efficiently on any device.
    """
    n_coils = calib_data.shape[0]
    spatial_shape = calib_data.shape[1:]
    n_dims = len(spatial_shape)

    # Use torch.Tensor.unfold to extract sliding windows
    # unfold works along one dimension at a time
    patches = calib_data
    for dim_idx in range(n_dims):
        patches = patches.unfold(dim_idx + 1, kernel_size[dim_idx], 1)

    # patches shape: (n_coils, *n_patches_per_dim, *kernel_size)
    n_patches_per_dim = [spatial_shape[i] - kernel_size[i] + 1 for i in range(n_dims)]
    n_patches = 1
    for n in n_patches_per_dim:
        n_patches *= n

    # Reshape to (n_patches, n_coils * kernel_elements)
    # First move coils after patch dims, then flatten
    if n_dims == 2:
        # patches: (C, p0, p1, k0, k1) -> (p0, p1, C, k0, k1)
        patches = patches.permute(1, 2, 0, 3, 4)
    elif n_dims == 3:
        # patches: (C, p0, p1, p2, k0, k1, k2) -> (p0, p1, p2, C, k0, k1, k2)
        patches = patches.permute(1, 2, 3, 0, 4, 5, 6)
    else:
        raise ValueError(f"Only 2D and 3D data supported, got {n_dims}D")

    cal_matrix = patches.reshape(n_patches, -1)
    return cal_matrix


def _compute_kernel_subspace(
    cal_matrix: torch.Tensor, threshold: float = 0.01
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find the signal-space kernels via eigendecomposition of the Gram matrix.

    Uses torch.linalg.eigh which works on CPU, CUDA, and MPS.
    """
    gram = cal_matrix.conj().T @ cal_matrix
    eigenvalues, eigenvectors = _eigh(gram)

    # Sort descending
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    svals = torch.sqrt(torch.abs(eigenvalues))
    max_sval = svals[0]
    if max_sval > 1e-12:
        sqrt_thresh = threshold**0.5
        n_keep = int((svals / max_sval > sqrt_thresh).sum().item())
    else:
        n_keep = 1
    n_keep = max(1, n_keep)

    kernels = eigenvectors[:, :n_keep].T.conj()
    return kernels, svals


def _transform_kernels_to_image_domain(
    kernels: torch.Tensor, kernel_size: tuple, n_coils: int
) -> torch.Tensor:
    """Zero-pad each k-space kernel and IFFT to obtain image-domain basis functions."""
    n_kernels = kernels.shape[0]
    n_dims = len(kernel_size)
    img_size = tuple(2 * k for k in kernel_size)
    starts = tuple((img_size[i] - kernel_size[i]) // 2 for i in range(n_dims))

    kernels_spatial = kernels.reshape(n_kernels, n_coils, *kernel_size)
    img_kernels = torch.zeros(
        (n_kernels, n_coils) + img_size,
        dtype=kernels.dtype,
        device=kernels.device,
    )

    if n_dims == 2:
        img_kernels[
            :,
            :,
            starts[0] : starts[0] + kernel_size[0],
            starts[1] : starts[1] + kernel_size[1],
        ] = kernels_spatial
        for k in range(n_kernels):
            for c in range(n_coils):
                img_kernels[k, c] = ifft2c(img_kernels[k, c])
    elif n_dims == 3:
        img_kernels[
            :,
            :,
            starts[0] : starts[0] + kernel_size[0],
            starts[1] : starts[1] + kernel_size[1],
            starts[2] : starts[2] + kernel_size[2],
        ] = kernels_spatial
        for k in range(n_kernels):
            for c in range(n_coils):
                img_kernels[k, c] = ifft3c(img_kernels[k, c])

    return img_kernels


def _compute_image_domain_covariance(
    img_kernels: torch.Tensor, kernel_size: tuple
) -> torch.Tensor:
    """
    Compute per-voxel Hermitian covariance H^H H from image-domain kernels.

    At each spatial position, the n_coils x n_coils matrix captures how signal
    energy correlates across coils. The dominant eigenvector IS the coil sensitivity.
    """
    n_dims = len(img_kernels.shape) - 2
    kernel_elements = 1
    for k in kernel_size:
        kernel_elements *= k
    img_elements = 1
    for s in img_kernels.shape[2:]:
        img_elements *= s
    normalization = kernel_elements / img_elements**2

    # Transpose to put spatial dims first: (*spatial, n_kernels, n_coils)
    axes_order = list(range(2, 2 + n_dims)) + [0, 1]
    H_t = img_kernels.permute(*axes_order)

    # Batched matrix multiply: H^H @ H
    # .contiguous() required — MPS gives wrong results with non-contiguous views
    H_conj_t = H_t.conj().transpose(-2, -1).contiguous()
    cov = torch.matmul(H_conj_t, H_t) / normalization
    return cov


# =============================================================================
# STEP 6: SINC-INTERPOLATE COVARIANCE + EXTRACT CSM
# =============================================================================


def _build_sinc_interpolation_matrix(
    n_in: int, n_out: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Precompute a sinc interpolation matrix of shape (n_out, n_in)."""
    if n_in == n_out:
        return torch.eye(n_in, dtype=dtype, device=device)

    M = torch.zeros((n_out, n_in), dtype=dtype, device=device)
    for j in range(n_in):
        e_j = torch.zeros(n_in, dtype=dtype, device=device)
        e_j[j] = 1.0
        tmp = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(e_j)))
        new_shape = torch.zeros(n_out, dtype=dtype, device=device)
        if n_out > n_in:
            start = (n_out - n_in) // 2
            new_shape[start : start + n_in] = tmp
        else:
            start = (n_in - n_out) // 2
            new_shape = tmp[start : start + n_out].clone()
        result = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(new_shape)))
        result = result * (n_out / n_in)
        M[:, j] = result
    return M


def _sinc_interp_axis(data: torch.Tensor, target_size: int, axis: int) -> torch.Tensor:
    """Sinc-interpolate data along one axis via FFT zero-padding."""
    source_size = data.shape[axis]
    if source_size == target_size:
        return data.clone()

    tmp = torch.fft.ifftshift(data, dim=axis)
    tmp = torch.fft.fft(tmp, dim=axis)
    tmp = torch.fft.fftshift(tmp, dim=axis)

    new_shape = list(tmp.shape)
    new_shape[axis] = target_size
    if target_size > source_size:
        padded = torch.zeros(new_shape, dtype=tmp.dtype, device=tmp.device)
        start = (target_size - source_size) // 2
        sl = [slice(None)] * tmp.ndim
        sl[axis] = slice(start, start + source_size)
        padded[tuple(sl)] = tmp
    else:
        start = (source_size - target_size) // 2
        sl = [slice(None)] * tmp.ndim
        sl[axis] = slice(start, start + target_size)
        padded = tmp[tuple(sl)].clone()

    result = torch.fft.ifftshift(padded, dim=axis)
    result = torch.fft.ifft(result, dim=axis)
    result = torch.fft.fftshift(result, dim=axis)
    result = result * (target_size / source_size)
    return result


def _run_power_iteration(
    cov: torch.Tensor, num_iter: int = 30
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched power iteration to find the dominant eigenvector.

    cov: (B, nc, nc) complex tensor
    Returns: (B, nc) eigenvectors, (B,) eigenvalues
    """
    B, nc, _ = cov.shape

    # Random complex initialisation (construct from real parts for MPS compat)
    v_real = torch.randn(B, nc, 1, device=cov.device)
    v_imag = torch.randn(B, nc, 1, device=cov.device)
    v = torch.complex(v_real, v_imag)
    v = v / (_complex_norm(v, dim=1, keepdim=True) + 1e-12)

    for _ in range(num_iter):
        v = torch.bmm(cov, v)
        v = v / (_complex_norm(v, dim=1, keepdim=True) + 1e-12)

    Av = torch.bmm(cov, v)
    eig_val = torch.sum(v.conj() * Av, dim=1).real.squeeze(-1)
    eig_vec = v.squeeze(-1)
    return eig_vec, eig_val


def _compute_eigenmaps_batched(
    img_cov: torch.Tensor, orthiter: bool = True, num_orthiter: int = 30
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the dominant eigenvector from a batch of per-voxel covariance matrices.

    img_cov: (*spatial, nc, nc)
    """
    nc = img_cov.shape[-1]
    spatial_shape = img_cov.shape[:-2]
    n_voxels = 1
    for s in spatial_shape:
        n_voxels *= s
    dtype = img_cov.dtype

    if orthiter:
        cov_flat = img_cov.reshape(n_voxels, nc, nc)
        all_vecs, all_vals = _run_power_iteration(cov_flat, num_iter=num_orthiter)

        all_vecs = all_vecs.reshape(*spatial_shape, nc)
        all_vals = all_vals.reshape(*spatial_shape)

        csm = torch.zeros((1, nc) + spatial_shape, dtype=dtype, device=img_cov.device)
        eigenvalues = torch.zeros((1,) + spatial_shape, device=img_cov.device)
        csm[0] = all_vecs.movedim(-1, 0)
        eigenvalues[0] = all_vals
    else:
        eigvals_all, eigvecs_all = _eigh(img_cov)
        csm = torch.zeros((1, nc) + spatial_shape, dtype=dtype, device=img_cov.device)
        eigenvalues = torch.zeros((1,) + spatial_shape, device=img_cov.device)
        csm[0] = eigvecs_all[..., -1].movedim(-1, 0)
        eigenvalues[0] = eigvals_all[..., -1]

    return csm, eigenvalues


def _interpolate_covariance_and_extract_csm(
    img_cov: torch.Tensor,
    target_shape: tuple,
    orthiter: bool = True,
    num_orthiter: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sinc-interpolate covariance to full resolution and extract eigenmaps.

    Uses slice-by-slice processing for 3D to manage memory.
    """
    n_dims = img_cov.ndim - 2
    nc = img_cov.shape[-1]
    device = img_cov.device
    dtype = img_cov.dtype

    if n_dims == 2:
        ny, nx = target_shape
        cov_full = _sinc_interp_axis(img_cov, ny, axis=0)
        cov_full = _sinc_interp_axis(cov_full, nx, axis=1)
        return _compute_eigenmaps_batched(cov_full, orthiter, num_orthiter)

    elif n_dims == 3:
        nz, ny, nx = target_shape
        nz_s, ny_s, nx_s = img_cov.shape[:3]

        # Pack upper triangle to halve memory
        cosize = nc * (nc + 1) // 2
        cov_packed = torch.zeros((nz_s, ny_s, nx_s, cosize), dtype=dtype, device=device)
        tri_i = torch.zeros(cosize, dtype=torch.long, device=device)
        tri_j = torch.zeros(cosize, dtype=torch.long, device=device)
        idx = 0
        for i in range(nc):
            for j in range(i + 1):
                cov_packed[..., idx] = img_cov[..., i, j]
                tri_i[idx] = i
                tri_j[idx] = j
                idx += 1

        # Precompute sinc interpolation matrices
        M_z = _build_sinc_interpolation_matrix(nz_s, nz, device, dtype)
        M_y = _build_sinc_interpolation_matrix(ny_s, ny, device, dtype)
        M_x = _build_sinc_interpolation_matrix(nx_s, nx, device, dtype)

        # Interpolate z for whole volume
        cov_z = (M_z @ cov_packed.reshape(nz_s, -1)).reshape(nz, ny_s, nx_s, cosize)

        csm = torch.zeros((1, nc, nz, ny, nx), dtype=dtype, device=device)
        eigenvalues = torch.zeros((1, nz, ny, nx), device=device)

        for z in range(nz):
            slc = (M_y @ cov_z[z].reshape(ny_s, -1)).reshape(ny, nx_s, cosize)
            slc = (
                (M_x @ slc.permute(1, 0, 2).reshape(nx_s, -1))
                .reshape(nx, ny, cosize)
                .permute(1, 0, 2)
            )

            # Reconstruct full Hermitian matrix from upper triangle
            cov_full = torch.zeros((ny, nx, nc, nc), dtype=dtype, device=device)
            cov_full[..., tri_i, tri_j] = slc
            cov_full[..., tri_j, tri_i] = slc.conj()

            s, e = _compute_eigenmaps_batched(cov_full, orthiter, num_orthiter)
            csm[:, :, z] = s
            eigenvalues[:, z] = e

        return csm, eigenvalues
    else:
        raise ValueError(f"Only 2D and 3D supported, got {n_dims}D")


# =============================================================================
# STEPS 7–9: POST-PROCESSING
# =============================================================================


def _mask_sensitivity_maps(
    csm: torch.Tensor,
    eigenvalues: torch.Tensor,
    mask_threshold: float = 0.8,
    soft_threshold: bool = False,
) -> torch.Tensor:
    """Zero out voxels with eigenvalues below the threshold."""
    dom_eigval = eigenvalues[0]
    if soft_threshold:
        weight = torch.sqrt(torch.abs(dom_eigval))
        weight = (weight - mask_threshold) / (1.0 - mask_threshold)

        result = torch.zeros_like(weight)
        mask_mid = weight >= -1
        mask_high = weight >= 1
        s = weight / (weight**2 + 1)
        result[mask_mid] = s[mask_mid]
        result[mask_high] += 1.0
        weight = result
    else:
        weight = (torch.abs(dom_eigval) >= mask_threshold).to(csm.real.dtype)

    csm = csm * weight.unsqueeze(0).unsqueeze(0)
    return csm


def _build_phase_rotation_matrix(calib_data: torch.Tensor) -> torch.Tensor:
    """Compute a unitary matrix that aligns coil phases to a common reference."""
    n_coils = calib_data.shape[0]
    calib_flat = calib_data.reshape(n_coils, -1)
    gram = calib_flat @ calib_flat.conj().T / calib_flat.shape[1]
    _, eigenvectors = _eigh(gram)
    # Sort descending by eigenvalue
    return eigenvectors.flip(-1)


def _apply_phase_rotation(
    csm: torch.Tensor, rotation_matrix: torch.Tensor
) -> torch.Tensor:
    """Remove the global phase ambiguity using the calibration-based rotation matrix."""
    num_maps, n_coils = csm.shape[0], csm.shape[1]
    csm_flat = csm.reshape(num_maps, n_coils, -1)
    ref_vec = rotation_matrix[:, 0].conj()
    for m in range(num_maps):
        # matmul instead of einsum — complex einsum is buggy on MPS
        phase_ref = (ref_vec.unsqueeze(0) @ csm_flat[m]).squeeze(0)
        phase_ref = phase_ref / (torch.abs(phase_ref) + 1e-10)
        csm_flat[m] = csm_flat[m] * phase_ref.conj().unsqueeze(0)
    return csm


def _normalize_sensitivity_maps(csm: torch.Tensor) -> torch.Tensor:
    """Normalise maps so RSS across coils = 1."""
    norm = torch.sqrt(torch.sum(torch.abs(csm) ** 2, dim=1, keepdim=True))
    norm = torch.where(norm > 1e-10, norm, torch.ones_like(norm))
    return csm / norm
