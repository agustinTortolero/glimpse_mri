# Glimpse MRI — Iteration 1 (Qt + CUDA, FFT-only)

**Goal:** Minimal MRI viewer for Windows (Qt 6) that reads *fastMRI* HDF5 raw data, reconstructs with **FFT + RSS** on GPU (CUDA/cuFFT), and lets the user **save PNG or DICOM** from a simple GUI (context menu).

> MVC layout:
> - **Model**: `mri_engine` — reconstruction (FFT-only in this iteration) using CUDA/cuFFT.
> - **View**: Qt Widgets UI (QLabel) + context menu for saving **PNG** and **DICOM**.
> - **Controller**: Orchestrates I/O (HDF5 fastMRI only) and calls into the model.

## Features (Iteration 1)
- Load **fastMRI** HDF5 (C × ky × kx) raw k-space (hardcoded path in this iteration).
- **GPU** IFFT (cuFFT) per coil + **RSS** coil combine.
- Standard **center-crop** to 320×320 (matches fastMRI published recon size).
- Display the image with automatic fit-to-window scaling.
- **Save as PNG** (OpenCV) and **Save as DICOM** (DCMTK).
- Console debug logs throughout (load, recon, scaling, save paths).

**Tested**
- Qt app displays **1 slice**
- Write to disk: **DICOM** and **PNG**

## Directory Layout
glimpse_dev_v1/
controller/
model/
src/
view/
build/ # (ignored)
glimpse_dev_v1.pro

## Dependencies (Windows)
- Qt 6.9 (MSVC 2022, x64)
- CUDA 12.4 (cuFFT)
- OpenCV 4.9 (opencv_world490[d].lib)
- HDF5 + DCMTK (via vcpkg)

## Build
Open `glimpse_dev_v1.pro` in Qt Creator → Desktop Qt 6.9.x MSVC2022 64bit → Debug build.
Copy runtime DLLs next to the EXE or ensure they’re on PATH.

## Run
On launch, loads the hardcoded fastMRI file:

C:\datasets\MRI_raw\FastMRI\brain_multicoil\file_brain_AXFLAIR_200_6002452.h5
Right-click image → **Save PNG…** or **Save DICOM…**.

## Roadmap
- File open dialog
- Window/level & tools
- ISMRMRD reader
- Slice navigation
- “Send to Gadgetron” (batch/stream)
