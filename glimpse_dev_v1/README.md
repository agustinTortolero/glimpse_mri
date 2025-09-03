Glimpse MRI

Glimpse MRI is a Windows High-Performance Computing (HPC) application for visualizing MRI scans and exporting them to common image formats. Built in modern C++/Qt and accelerated with NVIDIA CUDA, it supports imaging R&D teams working on MRI algorithms, hardware, and methodologies. Written in C++ with Qt 6, Glimpse MRI offloads compute-intensive MRI stages—coil noise pre-whitening, CG-SENSE reconstruction, and post-filtering—to NVIDIA GPUs via CUDA (cuFFT, cuBLAS, custom kernels). It reads DICOM and HDF5 (fastMRI/ISMRMRD), offers slice-wise inspection/export, and includes hooks for parameter sweeps and PSNR/SSIM-style QA to quantify improvements. Glimpse MRI is a Windows Model–View–Controller (MVC) application, built with Qt, for viewing and converting MRI images. It supports both DICOM and raw MRI data in fastMRI and ISMRMRD HDF5 formats, with CUDA-accelerated reconstruction on the GPU. Purpose: portfolio project to demonstrate HPC with CUDA, clear software architecture, and investigative skills (parameter experiments, algorithm comparisons, quantitative metrics). Note: For research/education only—not for clinical use.

Overview

Windows High-Performance Computing (HPC) application for visualizing MRI scans and exporting to common image formats.

Built in modern C++/Qt and accelerated with NVIDIA CUDA to support imaging R&D workflows (MRI algorithms, hardware, methodologies).

Offloads compute-intensive stages (coil noise pre-whitening, CG-SENSE reconstruction, post-filtering) to the GPU (cuFFT, cuBLAS, custom kernels).

Reads DICOM and HDF5 (fastMRI / ISMRMRD), offers slice-wise inspection/export, and provides hooks for parameter sweeps and PSNR/SSIM-style QA.

Purpose: research/education tool for algorithm exploration, parameter studies, and quantitative comparisons.

Note: For research/education only — not for clinical use.

Features

File support

DICOM images

HDF5 (.h5) datasets in fastMRI and ISMRMRD formats

Reconstruction and processing

Coil noise pre-whitening

CG-SENSE reconstruction

2D spatial filtering (extensible pipeline)

GPU acceleration via CUDA (cuFFT, cuBLAS, custom kernels)

Experimentation and QA

Parameter sweeps for algorithms

Hooks for PSNR/SSIM comparisons against references

Structured debug logs for reproducibility

User interface

Simple Qt GUI

Interactive zoom and slice navigation (Arrow keys + Ctrl)

Context menu to export PNG or BMP

Architecture

Back-end (Model) — mri_engine.dll

Reads .h5 (fastMRI / ISMRMRD)

Pre-processing: coil noise pre-whitening

Reconstruction: CG-SENSE

Post-processing: 2D spatial filters

All heavy compute on the GPU (CUDA)

Front-end (View)

Qt GUI for visualization

Displays reconstructed images

Tools: zoom, slice scrolling, export

Controller

Orchestrates Model <-> View

HDF5 (.h5): calls mri_engine -> GPU processing -> returns OpenCV cv::Mat -> converted to QImage for display

DICOM: uses standard C++ DICOM libraries to parse and display

Coordinates image saving/export

Tech Stack

Language: C++

Framework: Qt 6 (GUI)

GPU: CUDA (cuFFT, cuBLAS, custom kernels)

Imaging: OpenCV

Formats: DICOM, fastMRI, ISMRMRD

Roadmap

First release

DICOM, fastMRI, and ISMRMRD support

Slice visualization of MRI images

GPU acceleration of compute-intensive tasks

Saving as DICOM, PNG, and BMP

CG-SENSE

2D spatial median filtering

Pre-whitening using the algorithm in mri/prewhiten.hpp (uses mri/io.hpp and mri/common.hpp; Cholesky-based whitener from k-space corner covariance; CPU apply_whitener_cpu, CUDA path apply_whitener_cuda)

Second release

CPU multithreaded implementation

CG-SENSE-TV

ADD HERE A BETTER FILTERING ALGORITHM

Improved pre-whitening algorithm (e.g., shrinkage-regularized covariance / eigenvalue clipping)
