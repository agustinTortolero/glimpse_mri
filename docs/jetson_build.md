# Jetson Build Guide

This document records the validated Jetson build flow for **Glimpse MRI**.

## Status

Validated on Jetson with:

- aarch64
- CUDA available through `nvcc`
- CMake-based build flow
- successful build of:
  - `engine`
  - `dicom_io_lib`
  - `gui`

## Branch

Validated on:

`feature/jetson-cmake-port`

## One-command Jetson build

From the repository root:

```bash
./scripts/targets/jetson/build.sh

