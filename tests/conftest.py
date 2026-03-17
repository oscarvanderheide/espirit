"""Shared test fixtures and device parametrization."""

import pytest
import torch

from espirit import available_devices


def _device_ids():
    return [str(d) for d in available_devices()]


@pytest.fixture(params=available_devices(), ids=_device_ids())
def device(request):
    """Parametrized fixture that yields each available torch device."""
    return request.param


def make_synthetic_kspace_2d(
    n_coils: int = 4,
    ny: int = 64,
    nx: int = 64,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Create synthetic 2D multi-coil k-space with smooth coil sensitivities.

    Generates image-domain coil images with Gaussian sensitivity profiles,
    then FFTs to k-space. Useful for unit testing ESPIRiT pipeline steps.
    """
    device = device or torch.device("cpu")
    # Create a simple phantom (circle)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, ny, device=device),
        torch.linspace(-1, 1, nx, device=device),
        indexing="ij",
    )
    phantom = (yy**2 + xx**2 < 0.5**2).to(torch.complex64)

    # Create smooth coil sensitivities (shifted Gaussians)
    angles = torch.linspace(0, 2 * torch.pi, n_coils + 1, device=device)[:-1]
    coil_imgs = torch.zeros(n_coils, ny, nx, dtype=torch.complex64, device=device)
    for c in range(n_coils):
        cy, cx = 0.3 * torch.sin(angles[c]), 0.3 * torch.cos(angles[c])
        sensitivity = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 0.5)
        coil_imgs[c] = phantom * sensitivity.to(torch.complex64)

    # FFT to k-space
    axes = (-2, -1)
    kspace = torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(coil_imgs, dim=axes), dim=axes), dim=axes
    )
    return kspace


def make_synthetic_kspace_3d(
    n_coils: int = 4,
    nz: int = 16,
    ny: int = 32,
    nx: int = 32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Create synthetic 3D multi-coil k-space with smooth coil sensitivities.
    """
    device = device or torch.device("cpu")

    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, nz, device=device),
        torch.linspace(-1, 1, ny, device=device),
        torch.linspace(-1, 1, nx, device=device),
        indexing="ij",
    )
    phantom = (zz**2 + yy**2 + xx**2 < 0.4**2).to(torch.complex64)

    angles = torch.linspace(0, 2 * torch.pi, n_coils + 1, device=device)[:-1]
    coil_imgs = torch.zeros(n_coils, nz, ny, nx, dtype=torch.complex64, device=device)
    for c in range(n_coils):
        cy = 0.3 * torch.sin(angles[c])
        cx = 0.3 * torch.cos(angles[c])
        sensitivity = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 0.5)
        coil_imgs[c] = phantom * sensitivity.to(torch.complex64)

    axes = (-3, -2, -1)
    kspace = torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(coil_imgs, dim=axes), dim=axes), dim=axes
    )
    return kspace
