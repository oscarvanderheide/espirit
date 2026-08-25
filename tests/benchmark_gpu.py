"""Compare ESPIRiT GPU memory with compute-device and CPU output.

Usage:
    python tests/benchmark_gpu.py
    python tests/benchmark_gpu.py --shape 224 240 192 --ncoils 64
    python tests/benchmark_gpu.py --output-devices cpu
"""

import argparse
import sys
import time

import torch
import numpy as np

from espirit import espirit


def _make_phantom_kspace(n_coils, nz, ny, nx, device="cpu"):
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, nz, device=device),
        torch.linspace(-1, 1, ny, device=device),
        torch.linspace(-1, 1, nx, device=device),
        indexing="ij",
    )
    phantom = (zz**2 + yy**2 + xx**2 < 0.6**2).to(torch.complex64)

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
    return kspace.numpy()


def run_bench(kspace, output_device, calib_size, kernel_size, device):
    compute_device = torch.device(device)
    if compute_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(compute_device)

    start = time.time()
    csm = espirit(
        kspace,
        calib_size=calib_size,
        kernel_size=kernel_size,
        device=compute_device,
        output_device=None if output_device == "compute" else output_device,
    )
    if compute_device.type == "cuda":
        torch.cuda.synchronize(compute_device)
    elapsed = time.time() - start

    peak_gib = None
    if compute_device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(compute_device) / 1024**3

    rss = np.sqrt(np.sum(np.abs(csm) ** 2, axis=0))
    valid = rss > 0.1
    rss_mean = float(rss[valid].mean()) if valid.any() else 0

    return {
        "peak_gib": peak_gib,
        "runtime": elapsed,
        "rss_mean": rss_mean,
        "shape": csm.shape,
    }


def main():
    parser = argparse.ArgumentParser(description="ESPIRiT GPU memory benchmark")
    parser.add_argument("--shape", type=int, nargs=3, default=[224, 240, 192])
    parser.add_argument("--ncoils", type=int, default=64)
    parser.add_argument("--calib-size", type=int, default=24)
    parser.add_argument("--kernel-size", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-devices",
        nargs="+",
        default=["compute", "cpu"],
        choices=["compute", "cpu"],
        help="Where to accumulate the final CSM (default: both)",
    )
    args = parser.parse_args()

    nz, ny, nx = args.shape
    n_coils = args.ncoils
    print(f"Data: {n_coils} coils, shape ({nz}, {ny}, {nx})")
    print(f"Calib: {args.calib_size}, kernel: {args.kernel_size}")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print()

    kspace = _make_phantom_kspace(n_coils, nz, ny, nx, device="cpu")

    results = {}

    for output_device in args.output_devices:
        sys.stdout.write(f"  {output_device:30s} ... ")
        sys.stdout.flush()
        try:
            results[output_device] = run_bench(
                kspace,
                output_device,
                args.calib_size,
                args.kernel_size,
                args.device,
            )
            print("done")
        except Exception as exc:
            results[output_device] = f"FAILED: {exc}"
            print("failed")

    print(
        "\n{:<20s} {:>10s} {:>10s} {:>10s}".format(
            "Output", "Peak GiB", "Time (s)", "RSS mean"
        )
    )
    print("-" * 56)
    for label, r in results.items():
        if isinstance(r, dict):
            peak_str = f"{r['peak_gib']:.2f}" if r["peak_gib"] is not None else "N/A"
            print(
                f"{label:<20s} {peak_str:>10s} {r['runtime']:>10.1f} "
                f"{r['rss_mean']:>10.4f}"
            )
        else:
            print(f"{label:<20s} {r}")


if __name__ == "__main__":
    main()
