# Smoke Benchmarks — End-to-End Reconstruction Time

These are **quick smoke/perf sanity checks** for GlimpseMRI.  
They are **not** meant to be rigorous benchmarking.

**What’s included in the “total time” number**
- Disk → RAM (file read)
- Parsing / preprocessing
- RAM → GPU memory transfers (when applicable)
- Reconstruction compute
- GPU memory → RAM
- Any finalization before the app reports the reconstruction result

The value is taken from the log line:

`[CTRL][LOAD] LIB recon ok= true  ms= <N>`

---

## Systems Used

### Laptop (Windows)
Model: **MSI Cyborg 15 (A12V / A12VF series)**  
- **CPU:** Intel Core **i7-12650H**
- **GPU:** NVIDIA GeForce **RTX 4060 Laptop GPU (8GB)** *(nvidia-smi: 8188 MiB)*
- **RAM:** **32 GB DDR5**
- **CUDA Toolkit (nvcc):** `release 12.4, V12.4.131`
- **NVIDIA Driver:** `576.40`
- **QMake:** `3.1`
- **Qt:** `6.10.0` (`C:/Qt/6.10.0/msvc2022_64/lib`)

### Jetson
Device: **NVIDIA Jetson Orin Nano Developer Kit**  
Software stack used during the test:
- Jetson Linux / L4T: `R36 (release), REVISION: 4.4`
- Qt: `Qt 6.2.4` (apt)
- QMake: `3.1`
- CUDA: `12.6` (JetPack)

---

## Dataset Used + Attribution

File:
- `52c2fd53-d233-4444-8bfd-7c454240d314.h5`

Source:
- **MRIdata — “Stanford Fullysampled 3D FSE Knees”** dataset  
- Listed on MRIdata as part of that dataset group (uploader shown as **mikgroup**)

Links:
- MRIdata dataset listing: https://mridata.org/datasets
- MRIdata project: https://mridata.org/
- License/terms: https://mridata.org/terms
- HuggingFace mirror (shows license `cc-by-nc-4.0`): https://huggingface.co/datasets/arjundd/mridata-stanford-knee-3d-fse

Attribution:
> Dataset: MRIdata “Stanford Fullysampled 3D FSE Knees” (`52c2fd53-d233-4444-8bfd-7c454240d314.h5`), via mridata.org (uploader: mikgroup). License: CC BY-NC 4.0 (see MRIdata terms).

---


## Results
| System | GPU total recon time (ms) | CPU total recon time (ms) | Speedup (CPU/GPU) |
|---|---:|---:|---:|
| Laptop | 14020 | 17144 | 1.22× |
| Jetson Orin Nano | 78513 | 89817 | 1.14× |

> Notes:
> - 




