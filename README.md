# Glimpse MRI

Glimpse MRI is a high-performance MRI reconstruction and visualization application.

It combines a Qt-based desktop GUI with a C++ MRI engine that supports CPU and GPU acceleration depending on target platform.

## Build Guides

- [Jetson build guide](docs/jetson_build.md)
- [Ubuntu x86_64 build guide](docs/ubuntu_x86_64_build.md)
- [Raspberry Pi 5 ARM64 build guide](docs/rpi5_aarch64_build.md)

## What this repository includes

- `engine/` — MRI reconstruction engine
- `gui/` — desktop application (Qt)
- `dicom_io_lib/` — DICOM I/O support library
- `scripts/targets/` — target-specific dependency, build, and packaging scripts

## Supported data paths

- DICOM
- ISMRMRD (HDF5)
- fastMRI single-coil workflows

## Project status

The project is under active development; build flows and packaging may evolve.
