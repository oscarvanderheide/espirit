"""Centered FFT helpers using torch.fft (device-agnostic)."""

import torch


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered 2D FFT."""
    axes = (-2, -1)
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(x, dim=axes), dim=axes), dim=axes
    )


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered 2D IFFT."""
    axes = (-2, -1)
    return torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(x, dim=axes), dim=axes), dim=axes
    )


def fft3c(x: torch.Tensor) -> torch.Tensor:
    """Centered 3D FFT."""
    axes = (-3, -2, -1)
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(x, dim=axes), dim=axes), dim=axes
    )


def ifft3c(x: torch.Tensor) -> torch.Tensor:
    """Centered 3D IFFT."""
    axes = (-3, -2, -1)
    return torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(x, dim=axes), dim=axes), dim=axes
    )
