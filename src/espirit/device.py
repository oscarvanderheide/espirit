"""Device selection utilities for ESPIRiT."""

import torch


def select_device(preference: str | None = None) -> torch.device:
    """
    Select the best available torch device.

    Priority: cuda > mps > cpu (unless overridden by *preference*).

    Parameters
    ----------
    preference : str, optional
        Force a specific device ("cpu", "cuda", "mps").
        When None, auto-detects the best available.

    Returns
    -------
    torch.device
    """
    if preference is not None:
        return torch.device(preference)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def available_devices() -> list[torch.device]:
    """Return a list of all available torch devices (for testing)."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices
