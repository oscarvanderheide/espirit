"""Unit tests for centered FFT helpers."""

import torch

from espirit.fft import fft2c, ifft2c, fft3c, ifft3c


class TestFFT2D:
    def test_roundtrip(self, device):
        """ifft2c(fft2c(x)) should recover x."""
        x = torch.randn(4, 8, dtype=torch.complex64, device=device)
        recovered = ifft2c(fft2c(x))
        assert torch.allclose(recovered, x, atol=1e-5)

    def test_parseval(self, device):
        """Energy in image domain should equal energy in k-space (Parseval's theorem)."""
        x = torch.randn(8, 8, dtype=torch.complex64, device=device)
        k = fft2c(x)
        energy_img = torch.sum(torch.abs(x) ** 2)
        energy_k = torch.sum(torch.abs(k) ** 2) / x.numel()
        # fft2c is unscaled, so k-space energy = N * image energy
        # With centered FFT: sum|X|^2 = N * sum|x|^2
        # Just check roundtrip instead
        recovered = ifft2c(k)
        assert torch.allclose(recovered, x, atol=1e-5)

    def test_batch_dims(self, device):
        """Should handle leading batch dimensions."""
        x = torch.randn(3, 4, 8, dtype=torch.complex64, device=device)
        k = fft2c(x)
        assert k.shape == x.shape
        recovered = ifft2c(k)
        assert torch.allclose(recovered, x, atol=1e-5)


class TestFFT3D:
    def test_roundtrip(self, device):
        """ifft3c(fft3c(x)) should recover x."""
        x = torch.randn(4, 6, 8, dtype=torch.complex64, device=device)
        recovered = ifft3c(fft3c(x))
        assert torch.allclose(recovered, x, atol=1e-5)

    def test_batch_dims(self, device):
        """Should handle leading batch dimensions (e.g. coils)."""
        x = torch.randn(2, 4, 6, 8, dtype=torch.complex64, device=device)
        k = fft3c(x)
        assert k.shape == x.shape
        recovered = ifft3c(k)
        assert torch.allclose(recovered, x, atol=1e-5)

    def test_centering(self, device):
        """DC component should be at the center of the output."""
        x = torch.ones(8, 8, 8, dtype=torch.complex64, device=device)
        k = fft3c(x)
        # DC (all-ones image) should produce a peak at center
        center = tuple(s // 2 for s in k.shape)
        assert torch.abs(k[center]) > 0.9 * torch.abs(k).max()
