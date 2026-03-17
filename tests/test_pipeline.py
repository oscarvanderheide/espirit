"""
Integration tests: full ESPIRiT pipeline consistency across all available devices.

Tests that:
1. The full pipeline runs without error on each device
2. Output shape and dtype are correct
3. Results are consistent across devices (CPU vs GPU vs MPS)
4. NumPy input/output works correctly
"""

import torch
import numpy as np
import pytest

from espirit import espirit, available_devices
from conftest import make_synthetic_kspace_2d, make_synthetic_kspace_3d


class TestFullPipeline2D:
    """Full ESPIRiT pipeline tests on 2D synthetic data."""

    def test_runs_and_returns_correct_shape(self, device):
        n_coils, ny, nx = 4, 64, 64
        kspace = make_synthetic_kspace_2d(n_coils=n_coils, ny=ny, nx=nx, device=device)
        csm = espirit(kspace, calib_size=12, kernel_size=6, device=device)
        assert csm.shape == (n_coils, ny, nx)
        assert csm.dtype == torch.complex64

    def test_numpy_input_output(self):
        """When given a NumPy array, should return a NumPy array."""
        n_coils, ny, nx = 4, 64, 64
        kspace_np = make_synthetic_kspace_2d(n_coils=n_coils, ny=ny, nx=nx).numpy()
        csm = espirit(kspace_np, calib_size=12, kernel_size=6, device="cpu")
        assert isinstance(csm, np.ndarray)
        assert csm.shape == (n_coils, ny, nx)
        assert csm.dtype == np.complex64

    def test_normalized_rss(self, device):
        """RSS of output should be ~1 in the support region."""
        kspace = make_synthetic_kspace_2d(n_coils=4, ny=64, nx=64, device=device)
        csm = espirit(kspace, calib_size=12, kernel_size=6, device=device)
        rss = torch.sqrt(torch.sum(torch.abs(csm) ** 2, dim=0))
        # In the support region (where eigenvalue mask passes), RSS should be ~1
        supported = rss > 0.5
        if supported.any():
            assert torch.allclose(
                rss[supported], torch.ones_like(rss[supported]), atol=0.05
            )


class TestFullPipeline3D:
    """Full ESPIRiT pipeline tests on 3D synthetic data."""

    def test_runs_and_returns_correct_shape(self, device):
        n_coils, nz, ny, nx = 4, 16, 32, 32
        kspace = make_synthetic_kspace_3d(
            n_coils=n_coils, nz=nz, ny=ny, nx=nx, device=device
        )
        csm = espirit(kspace, calib_size=8, kernel_size=4, device=device)
        assert csm.shape == (n_coils, nz, ny, nx)
        assert csm.dtype == torch.complex64

    def test_numpy_roundtrip(self):
        n_coils, nz, ny, nx = 4, 16, 32, 32
        kspace_np = make_synthetic_kspace_3d(
            n_coils=n_coils, nz=nz, ny=ny, nx=nx
        ).numpy()
        csm = espirit(kspace_np, calib_size=8, kernel_size=4, device="cpu")
        assert isinstance(csm, np.ndarray)
        assert csm.shape == (n_coils, nz, ny, nx)


class TestCrossDeviceConsistency:
    """Verify that different devices produce the same result."""

    @pytest.fixture()
    def reference_csm_2d(self):
        """CPU reference result for 2D."""
        torch.manual_seed(42)
        kspace = make_synthetic_kspace_2d(n_coils=4, ny=64, nx=64)
        return espirit(
            kspace,
            calib_size=12,
            kernel_size=6,
            device="cpu",
            orthiter=False,  # Use eigh for deterministic results
        )

    @pytest.fixture()
    def reference_csm_3d(self):
        """CPU reference result for 3D."""
        torch.manual_seed(42)
        kspace = make_synthetic_kspace_3d(n_coils=4, nz=16, ny=32, nx=32)
        return espirit(
            kspace,
            calib_size=8,
            kernel_size=4,
            device="cpu",
            orthiter=False,
        )

    def test_consistency_2d(self, reference_csm_2d):
        """All devices should produce equivalent 2D CSMs."""
        ref = reference_csm_2d
        for dev in available_devices():
            if dev.type == "cpu":
                continue
            torch.manual_seed(42)
            # Generate on CPU (for reproducibility) then move
            kspace = make_synthetic_kspace_2d(n_coils=4, ny=64, nx=64).to(dev)
            csm = espirit(
                kspace,
                calib_size=12,
                kernel_size=6,
                device=dev,
                orthiter=False,
            ).cpu()
            # Compare magnitudes (phase is arbitrary)
            assert torch.allclose(csm.abs(), ref.abs(), atol=0.1), (
                f"2D CSM magnitude mismatch on {dev}"
            )

    def test_consistency_3d(self, reference_csm_3d):
        """All devices should produce equivalent 3D CSMs."""
        ref = reference_csm_3d
        for dev in available_devices():
            if dev.type == "cpu":
                continue
            torch.manual_seed(42)
            kspace = make_synthetic_kspace_3d(n_coils=4, nz=16, ny=32, nx=32).to(dev)
            csm = espirit(
                kspace,
                calib_size=8,
                kernel_size=4,
                device=dev,
                orthiter=False,
            ).cpu()
            assert torch.allclose(csm.abs(), ref.abs(), atol=0.1), (
                f"3D CSM magnitude mismatch on {dev}"
            )


class TestOptions:
    """Test different option combinations don't crash."""

    @pytest.mark.parametrize("orthiter", [True, False])
    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("rotphase", [True, False])
    def test_option_combos_2d(self, device, orthiter, normalize, rotphase):
        kspace = make_synthetic_kspace_2d(n_coils=4, ny=32, nx=32, device=device)
        csm = espirit(
            kspace,
            calib_size=12,
            kernel_size=6,
            device=device,
            orthiter=orthiter,
            normalize=normalize,
            rotphase=rotphase,
        )
        assert csm.shape == (4, 32, 32)
        assert not torch.any(torch.isnan(csm))
