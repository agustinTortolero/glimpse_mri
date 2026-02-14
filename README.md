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

### Build from Source

#### 1) Clone the repository

```bash
git clone https://github.com/agustinTortolero/glimpse_mri.git
cd glimpse_mri
```

---

### Jetson (Orin Nano / JetPack / Ubuntu)

This repository includes scripts to install dependencies and build the full project (engine + GUI) on Jetson.

#### 2) One-time prerequisites

```bash
chmod +x prerequisites.sh build_jetson.shh
./prerequisites.sh
```

#### 3) Build (engine + GUI)

First build:

```bash
./build_jetson.shh --clean
```

Incremental rebuild (after `git pull`):

```bash
./build_jetson.shh
```

#### 4) Run

```bash
./build_gui_Release/glimpseMRI
```

#### Optional: create a clickable desktop icon

You can create a desktop/app-menu launcher after a successful build:

```bash
./build_jetson.shh --install-desktop
```

> On some GNOME setups you may need to right-click the desktop icon and choose **“Allow Launching”** once.

---


