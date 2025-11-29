# Glimpse MRI

Glimpse MRI is a high-performance MRI reconstruction application with image viewing and writing capabilities. It supports DICOM, ISMRMRD (HDF5), and fastMRI datasets.

Built with C++, CUDA, and Qt, Glimpse MRI delivers a clean, responsive interface while offloading heavy computation to a custom high-performance, heterogeneous MRI engine. The engine targets modern CPUs and GPUs using OpenMP for multi-threading, FFTW for CPU-side Fourier transforms, and custom CUDA kernels plus NVIDIA cuFFT and NVIDIA cuBLAS for GPU acceleration, enabling fast reconstruction and smooth, real-time interaction.

Developed by **Agustín Tortolero**.

---

## Features

- MRI image viewer with window/level control and slice navigation  
- MRI reconstruction engine targeting multi-core CPUs and NVIDIA GPUs  
- Support for:
  - DICOM series (via a dedicated DICOM I/O library)
  - ISMRMRD (HDF5) raw MRI acquisitions
  - fastMRI single-coil datasets
- Cross-platform GUI built with Qt 6 (Windows primary target; Linux planned)
- High-performance reconstruction kernels using FFTs and BLAS operations

> **Note:** Glimpse MRI is under active development. Interfaces, file formats, and performance characteristics may change between releases.

---

## Core Technologies

- **Languages & frameworks**
  - C++ (modern C++ for core engine and GUI)
  - CUDA for GPU-accelerated MRI reconstruction
  - Qt 6 for the cross-platform GUI

- **Parallelism & math libraries**
  - OpenMP for multi-threaded CPU execution
  - FFTW (Fastest Fourier Transform in the West) for CPU-side FFTs
  - NVIDIA cuFFT and NVIDIA cuBLAS for GPU FFTs and BLAS operations

---

## Data Formats & Imaging Libraries

- **DICOM**
  - Via a dedicated DICOM I/O library (separate module)
- **Raw MRI**
  - ISMRMRD (HDF5) for k-space acquisitions
- **Research datasets**
  - fastMRI single-coil datasets
- **Supporting libraries**
  - HDF5 and HDF5 C++ APIs
  - pugixml for XML metadata handling

---

## Development Environment

Glimpse MRI is developed and tested primarily on Windows with the following tools:

- Microsoft Visual Studio 2022 (MSVC toolchain, engine build)
- Qt Creator (Qt 6 project, GUI build and deployment)
- qmake for build configuration and project orchestration

Linux support is planned as part of future work.

---

## Getting Started

### Download

Pre-built Windows binaries (when available) can be found on the  
➡️ **[Releases page](../../releases)**

### Build from Source (high-level overview)

A detailed build guide is planned. At a high level, you will need:

- Microsoft Visual Studio 2022
- Qt 6 (matching your MSVC toolchain)
- CUDA Toolkit (for GPU acceleration)
- FFTW, cuFFT, cuBLAS, HDF5, and other third-party dependencies

Once dependencies are installed and configured, you can:

1. Clone the repository:
   ```bash
   git clone https://github.com/agustinTortolero/glimpse_mri.git
   cd glimpse_mri

