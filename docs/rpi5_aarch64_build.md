# Raspberry Pi 5 ARM64 Build Guide

This guide describes the CPU-only build and packaging flow for **Glimpse MRI** on Raspberry Pi 5 (aarch64).

## Scope

Target scripts are located in:

```bash
scripts/targets/rpi5_aarch64/
```

This flow is designed for Raspberry Pi OS / Ubuntu ARM64 environments on Pi 5 class hardware.

## 1) Install build prerequisites (Dev-Raspi)

From the repository root:

```bash
chmod +x scripts/targets/rpi5_aarch64/*.sh
sudo ./scripts/targets/rpi5_aarch64/prerequisites_build.sh
```

## 2) Build (Dev-Raspi)

```bash
./scripts/targets/rpi5_aarch64/build.sh
```

## 3) Package (Dev-Raspi)

```bash
./scripts/targets/rpi5_aarch64/pack.sh
```

This creates a tarball artifact for transfer to a test/runtime device.

## 4) Install runtime dependencies (Test-Raspi)

On the runtime Pi:

```bash
sudo ./scripts/targets/rpi5_aarch64/prerequisites_run.sh
```

## 5) Install and run from tarball (Test-Raspi)

```bash
tar -xzf glimpse_mri_aarch64_Release_*.tar.gz
cd glimpse_mri_aarch64_Release_*
./install.sh
~/glimpse_mri/run.sh
```

## Notes

- This target is **CPU-only** (`CUDA_MODE=off` in target flow).
- Packaging is tarball-based.
- `install.sh` supports `--prefix /opt/glimpse_mri` for system-wide install locations.
