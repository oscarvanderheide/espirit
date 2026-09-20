"""Unit tests for eigenmap extraction (step 6) and power iteration."""

import pytest
import torch

import sys

from espirit.espirit import (
    _eigh,
    _run_power_iteration,
    _compute_eigenmaps_batched,
    _build_sinc_interpolation_matrix,
    _sinc_interp_axis,
)


class TestPowerIteration:
    def test_recovers_dominant_eigenvector(self, device):
        """Power iteration should find the dominant eigenvector of a known matrix."""
        # Construct a Hermitian matrix with known dominant eigenvector
        nc = 4
        # Build on CPU and move to device (for MPS complex randn compat)
        v_true = torch.randn(nc, dtype=torch.complex64)
        v_true = v_true / torch.sqrt(torch.sum(torch.abs(v_true) ** 2))
        # Matrix with dominant eigenvalue 5, rest 1
        cov = torch.eye(nc, dtype=torch.complex64) + 4.0 * torch.outer(
            v_true, v_true.conj()
        )
        cov_batch = cov.unsqueeze(0).to(device)  # (1, nc, nc)
        v_true = v_true.to(device)

        vecs, vals = _run_power_iteration(cov_batch, num_iter=50)

        # Eigenvalue should be close to 5
        assert torch.abs(vals[0] - 5.0) < 0.1, f"Expected ~5.0, got {vals[0]:.3f}"

        # Eigenvector should be parallel to v_true (up to global phase)
        overlap = torch.abs(torch.dot(vecs[0].conj(), v_true))
        assert overlap > 0.99, f"Expected overlap > 0.99, got {overlap:.4f}"

    def test_batch(self, device):
        """Should handle multiple matrices in parallel."""
        B, nc = 16, 4
        cov = (
            (torch.eye(nc, dtype=torch.complex64) * 2)
            .expand(B, -1, -1)
            .contiguous()
            .to(device)
        )
        vecs, vals = _run_power_iteration(cov, num_iter=30)
        assert vecs.shape == (B, nc)
        assert vals.shape == (B,)
        # All eigenvalues should be ~2 for identity * 2
        assert torch.allclose(vals.cpu(), torch.full((B,), 2.0), atol=0.2)


class TestComputeEigenmapsBatched:
    def test_shape_2d(self, device):
        nc = 4
        # Fake covariance: (ny, nx, nc, nc)
        cov = (
            torch.eye(nc, dtype=torch.complex64)
            .expand(8, 8, -1, -1)
            .contiguous()
            .to(device)
        )
        csm, eigenvalues = _compute_eigenmaps_batched(
            cov, orthiter=True, num_orthiter=30
        )
        assert csm.shape == (1, nc, 8, 8)
        assert eigenvalues.shape == (1, 8, 8)

    def test_orthiter_vs_eigh(self, device):
        """Power iteration and full eigh should give similar dominant eigenvectors."""
        nc = 4
        # Create a meaningful covariance on CPU then move
        H = torch.randn(8, 8, 6, nc, dtype=torch.complex64)
        cov = torch.matmul(H.conj().transpose(-2, -1), H) / 6
        cov = cov.to(device)

        csm_orth, evals_orth = _compute_eigenmaps_batched(
            cov, orthiter=True, num_orthiter=50
        )
        csm_eigh, evals_eigh = _compute_eigenmaps_batched(cov, orthiter=False)

        # Eigenvalues should be similar
        assert torch.allclose(evals_orth.cpu(), evals_eigh.cpu(), rtol=0.05, atol=0.01)

        # Eigenvectors should be parallel (up to phase)
        overlap = torch.abs(torch.sum(csm_orth.conj() * csm_eigh, dim=1))
        assert (overlap.cpu() > 0.95).all(), f"Min overlap: {overlap.min():.4f}"


class TestChunkedEigh:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only code path")
    def test_chunked_matches_unchunked(self, monkeypatch):
        """Chunking a large CUDA batch must reproduce the plain eigh result."""
        nc = 5
        H = torch.randn(6, 7, 9, nc, dtype=torch.complex64, device="cuda")
        cov = torch.matmul(H.conj().transpose(-2, -1), H)  # (6, 7, nc, nc) batch of 42
        w_ref, v_ref = torch.linalg.eigh(cov)
        # `espirit.espirit` is shadowed by the re-exported function; go via sys.modules.
        monkeypatch.setattr(sys.modules["espirit.espirit"], "_EIGH_MAX_CUDA_BATCH", 10)
        w, v = _eigh(cov)
        assert w.shape == w_ref.shape and v.shape == v_ref.shape
        assert torch.allclose(w, w_ref, rtol=1e-4, atol=1e-5)
        # Eigenvectors agree up to a per-vector phase.
        overlap = torch.abs(torch.sum(v.conj() * v_ref, dim=-2))
        assert (overlap > 0.999).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only code path")
    def test_full_slice_batch(self):
        """A 224x224 slice of 13x13 matrices exceeds the cuSOLVER batch limit."""
        nc = 13
        H = torch.randn(224, 224, 20, nc, dtype=torch.complex64, device="cuda")
        cov = torch.matmul(H.conj().transpose(-2, -1), H)
        w, v = _eigh(cov)
        top = v[..., -1:]
        residual = (cov @ top - w[..., -1:, None] * top).abs().amax()
        assert residual / w.abs().max() < 1e-4


class TestSincInterpolation:
    def test_identity_when_same_size(self, device):
        M = _build_sinc_interpolation_matrix(8, 8, device, torch.complex64)
        assert torch.allclose(
            M, torch.eye(8, dtype=torch.complex64, device=device), atol=1e-5
        )

    def test_upsample_preserves_dc(self, device):
        """A constant signal should stay constant after sinc interpolation."""
        data = torch.ones(8, dtype=torch.complex64, device=device)
        upsampled = _sinc_interp_axis(data.unsqueeze(-1), 16, axis=0).squeeze(-1)
        # Should be approximately constant
        assert torch.allclose(upsampled.real, torch.ones(16, device=device), atol=0.2)

    @pytest.mark.parametrize("source_size,target_size", [(5, 9), (6, 10), (9, 5)])
    def test_fft_matches_interpolation_matrix(self, device, source_size, target_size):
        data = torch.randn(2, source_size, 3, dtype=torch.complex64, device=device)
        matrix = _build_sinc_interpolation_matrix(
            source_size, target_size, device, torch.complex64
        )
        expected = torch.einsum("oi,aib->aob", matrix, data)
        actual = _sinc_interp_axis(data, target_size, axis=1)
        assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_downsample_shape(self, device):
        data = torch.randn(4, 16, 3, dtype=torch.complex64, device=device)
        result = _sinc_interp_axis(data, 8, axis=1)
        assert result.shape == (4, 8, 3)
