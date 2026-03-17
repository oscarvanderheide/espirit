"""Unit tests for post-processing steps 7-9."""

import torch

from espirit.espirit import (
    _mask_sensitivity_maps,
    _build_phase_rotation_matrix,
    _apply_phase_rotation,
    _normalize_sensitivity_maps,
)


class TestMaskSensitivityMaps:
    def test_hard_mask(self, device):
        """Voxels below threshold should be zeroed out."""
        nc = 4
        csm = torch.ones(1, nc, 8, 8, dtype=torch.complex64, device=device)
        eigenvalues = torch.zeros(1, 8, 8, device=device)
        eigenvalues[:, :4, :] = 1.0  # top half above threshold
        eigenvalues[:, 4:, :] = 0.1  # bottom half below threshold

        masked = _mask_sensitivity_maps(csm, eigenvalues, mask_threshold=0.8)
        # Top half should be preserved
        assert masked[:, :, :4, :].abs().sum() > 0
        # Bottom half should be zeroed
        assert masked[:, :, 4:, :].abs().sum() == 0

    def test_soft_mask_no_crash(self, device):
        nc = 4
        csm = torch.ones(1, nc, 8, 8, dtype=torch.complex64, device=device)
        eigenvalues = torch.rand(1, 8, 8, device=device)
        masked = _mask_sensitivity_maps(
            csm, eigenvalues, mask_threshold=0.5, soft_threshold=True
        )
        assert masked.shape == csm.shape


class TestPhaseRotation:
    def test_builds_rotation_matrix(self, device):
        """Phase rotation matrix should have correct shape."""
        calib = torch.randn(4, 8, 8, dtype=torch.complex64, device=device)
        rot = _build_phase_rotation_matrix(calib)
        assert rot.shape == (4, 4)

    def test_apply_preserves_shape(self, device):
        calib = torch.randn(4, 8, 8, dtype=torch.complex64, device=device)
        rot = _build_phase_rotation_matrix(calib)
        csm = torch.randn(1, 4, 8, 8, dtype=torch.complex64, device=device)
        result = _apply_phase_rotation(csm, rot)
        assert result.shape == csm.shape


class TestNormalization:
    def test_rss_equals_one(self, device):
        """After normalization, RSS across coils should be ~1."""
        nc = 4
        csm = torch.randn(1, nc, 8, 8, dtype=torch.complex64, device=device)
        # Make sure there is signal
        csm = csm + 1.0
        normed = _normalize_sensitivity_maps(csm)
        rss = torch.sqrt(torch.sum(torch.abs(normed) ** 2, dim=1))
        assert torch.allclose(rss, torch.ones_like(rss), atol=1e-5)

    def test_zero_voxels_handled(self, device):
        """Voxels with zero signal should not produce NaN/Inf."""
        nc = 4
        csm = torch.zeros(1, nc, 4, 4, dtype=torch.complex64, device=device)
        normed = _normalize_sensitivity_maps(csm)
        assert not torch.any(torch.isnan(normed))
        assert not torch.any(torch.isinf(normed))
