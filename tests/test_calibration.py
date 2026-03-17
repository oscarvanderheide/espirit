"""Unit tests for ESPIRiT calibration steps 1-5."""

import torch

from espirit.espirit import (
    _extract_calibration_region,
    _build_calibration_matrix,
    _compute_kernel_subspace,
    _transform_kernels_to_image_domain,
    _compute_image_domain_covariance,
)
from conftest import make_synthetic_kspace_2d, make_synthetic_kspace_3d


class TestExtractCalibrationRegion:
    def test_shape_2d(self, device):
        kspace = make_synthetic_kspace_2d(n_coils=4, ny=64, nx=64, device=device)
        calib = _extract_calibration_region(kspace, (24, 24))
        assert calib.shape == (4, 24, 24)

    def test_shape_3d(self, device):
        kspace = make_synthetic_kspace_3d(n_coils=4, nz=16, ny=32, nx=32, device=device)
        calib = _extract_calibration_region(kspace, (8, 12, 12))
        assert calib.shape == (4, 8, 12, 12)

    def test_centered(self, device):
        """Calibration region should be centered in k-space."""
        kspace = torch.zeros(2, 32, 32, dtype=torch.complex64, device=device)
        kspace[:, 15:17, 15:17] = 1.0  # energy at center
        calib = _extract_calibration_region(kspace, (8, 8))
        # The center of kspace should be in the calibration region
        assert calib.abs().sum() > 0

    def test_device_preserved(self, device):
        kspace = make_synthetic_kspace_2d(device=device)
        calib = _extract_calibration_region(kspace, (12, 12))
        assert calib.device == kspace.device


class TestBuildCalibrationMatrix:
    def test_shape_2d(self, device):
        kspace = make_synthetic_kspace_2d(n_coils=4, ny=32, nx=32, device=device)
        calib = _extract_calibration_region(kspace, (12, 12))
        cal_matrix = _build_calibration_matrix(calib, (6, 6))
        n_patches = (12 - 6 + 1) ** 2  # 7 * 7 = 49
        kernel_elements = 4 * 6 * 6  # n_coils * kx * ky = 144
        assert cal_matrix.shape == (n_patches, kernel_elements)

    def test_shape_3d(self, device):
        kspace = make_synthetic_kspace_3d(n_coils=4, nz=16, ny=32, nx=32, device=device)
        calib = _extract_calibration_region(kspace, (8, 12, 12))
        cal_matrix = _build_calibration_matrix(calib, (4, 6, 6))
        n_patches = (8 - 4 + 1) * (12 - 6 + 1) * (12 - 6 + 1)
        kernel_elements = 4 * 4 * 6 * 6
        assert cal_matrix.shape == (n_patches, kernel_elements)

    def test_device_preserved(self, device):
        kspace = make_synthetic_kspace_2d(device=device)
        calib = _extract_calibration_region(kspace, (12, 12))
        cal_matrix = _build_calibration_matrix(calib, (6, 6))
        assert cal_matrix.device == kspace.device


class TestComputeKernelSubspace:
    def test_returns_kernels_and_svals(self, device):
        kspace = make_synthetic_kspace_2d(n_coils=4, ny=32, nx=32, device=device)
        calib = _extract_calibration_region(kspace, (12, 12))
        cal_matrix = _build_calibration_matrix(calib, (6, 6))
        kernels, svals = _compute_kernel_subspace(cal_matrix, threshold=0.001)
        assert kernels.ndim == 2
        assert kernels.shape[1] == cal_matrix.shape[1]
        assert svals.ndim == 1
        assert kernels.shape[0] >= 1  # at least one kernel retained

    def test_svals_descending(self, device):
        kspace = make_synthetic_kspace_2d(device=device)
        calib = _extract_calibration_region(kspace, (12, 12))
        cal_matrix = _build_calibration_matrix(calib, (6, 6))
        kernels, svals = _compute_kernel_subspace(cal_matrix, threshold=0.001)
        # Eigenvalues (and thus svals) from _eigh are sorted ascending,
        # then we reverse to descending. However sqrt(abs()) can reorder
        # near-zero/negative eigenvalues. Just ensure the top svals are descending.
        n_check = min(10, len(svals))
        for i in range(n_check - 1):
            assert svals[i] >= svals[i + 1] - 0.01

    def test_threshold_controls_kernel_count(self, device):
        kspace = make_synthetic_kspace_2d(n_coils=4, device=device)
        calib = _extract_calibration_region(kspace, (12, 12))
        cal_matrix = _build_calibration_matrix(calib, (6, 6))
        kernels_loose, _ = _compute_kernel_subspace(cal_matrix, threshold=0.0001)
        kernels_tight, _ = _compute_kernel_subspace(cal_matrix, threshold=0.1)
        assert kernels_loose.shape[0] >= kernels_tight.shape[0]


class TestTransformKernelsToImageDomain:
    def test_output_shape_2d(self, device):
        kspace = make_synthetic_kspace_2d(n_coils=4, device=device)
        calib = _extract_calibration_region(kspace, (12, 12))
        cal_matrix = _build_calibration_matrix(calib, (6, 6))
        kernels, _ = _compute_kernel_subspace(cal_matrix, threshold=0.001)
        img_kernels = _transform_kernels_to_image_domain(kernels, (6, 6), 4)
        assert img_kernels.shape[1] == 4  # n_coils
        assert img_kernels.shape[2:] == (12, 12)  # 2 * kernel_size

    def test_output_shape_3d(self, device):
        kspace = make_synthetic_kspace_3d(n_coils=4, device=device)
        calib = _extract_calibration_region(kspace, (8, 12, 12))
        cal_matrix = _build_calibration_matrix(calib, (4, 6, 6))
        kernels, _ = _compute_kernel_subspace(cal_matrix, threshold=0.001)
        img_kernels = _transform_kernels_to_image_domain(kernels, (4, 6, 6), 4)
        assert img_kernels.shape[1] == 4
        assert img_kernels.shape[2:] == (8, 12, 12)


class TestComputeImageDomainCovariance:
    def test_hermitian_2d(self, device):
        """Covariance should be Hermitian at each voxel."""
        kspace = make_synthetic_kspace_2d(n_coils=4, device=device)
        calib = _extract_calibration_region(kspace, (12, 12))
        cal_matrix = _build_calibration_matrix(calib, (6, 6))
        kernels, _ = _compute_kernel_subspace(cal_matrix)
        img_kernels = _transform_kernels_to_image_domain(kernels, (6, 6), 4)
        cov = _compute_image_domain_covariance(img_kernels, (6, 6))
        # Check Hermitian: C == C^H
        cov_H = cov.conj().transpose(-2, -1)
        assert torch.allclose(cov, cov_H, atol=1e-4)

    def test_shape(self, device):
        kspace = make_synthetic_kspace_2d(n_coils=4, device=device)
        calib = _extract_calibration_region(kspace, (12, 12))
        cal_matrix = _build_calibration_matrix(calib, (6, 6))
        kernels, _ = _compute_kernel_subspace(cal_matrix)
        img_kernels = _transform_kernels_to_image_domain(kernels, (6, 6), 4)
        cov = _compute_image_domain_covariance(img_kernels, (6, 6))
        # (2*ky, 2*kx, nc, nc)
        assert cov.shape == (12, 12, 4, 4)
