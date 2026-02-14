# Build GlimpseMRI on NVIDIA Jetson Orin Nano (JetPack / Ubuntu)

This is a **tested, minimal build recipe** for **GlimpseMRI** on **NVIDIA Jetson Orin Nano** using the repo:

- `https://github.com/agustinTortolero/glimpse_mri`

It assumes:
- Ubuntu 22.04 on Jetson (L4T R36 / JetPack 6.x)
- Qt 6 from apt (`qmake6`, `qt6-base-dev`, `qt6-multimedia-dev`)
- CUDA Toolkit from JetPack (`/usr/local/cuda`)

> Tip: run commands from a terminal on the Jetson desktop for GUI runs (or use `ssh -X` / `QT_QPA_PLATFORM=offscreen` for headless testing).

## Tested on

These instructions were verified on the following Jetson environment:

```bash
uname -a
cat /etc/nv_tegra_release
qmake6 -v
nvcc --version
```

Output:

```text
Linux agustin-jetson 5.15.148-tegra #1 SMP PREEMPT Mon Jun 16 08:24:48 PDT 2025 aarch64 aarch64 aarch64 GNU/Linux
# R36 (release), REVISION: 4.4, GCID: 41062509, BOARD: generic, EABI: aarch64, DATE: Mon Jun 16 16:07:13 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
QMake version 3.1
Using Qt version 6.2.4 in /usr/lib/aarch64-linux-gnu
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Aug_14_10:14:07_PDT_2024
Cuda compilation tools, release 12.6, V12.6.68
Build cuda_12.6.r12.6/compiler.34714021_0
```


## Where to place this file in the repo

Recommended path in the repo:

- `docs/build_jetson_orin_nano.md`

If you add it there, you can link it from your main `README.md` under a **Jetson** section.

---

## 0) JetPack + CUDA sanity

Install JetPack meta-package (includes CUDA, cuFFT, etc):

```bash
set -euxo pipefail
sudo apt update
sudo apt install -y nvidia-jetpack
```

Ensure `nvcc` is on PATH:

```bash
set -euxo pipefail
export PATH="/usr/local/cuda/bin:${PATH}"
hash -r
command -v nvcc
nvcc --version
```

(Optional) Persist CUDA PATH in `~/.bashrc`:

```bash
cat >> ~/.bashrc <<'EOF'

# --- CUDA (Jetson) ---
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
EOF
```

If you use `set -u` and your `.bashrc` errors on `force_color_prompt`, patch it:

```bash
perl -pi -e 's/\$force_color_prompt/\${force_color_prompt:-}/g' ~/.bashrc
```

---

## 1) Install build dependencies

```bash
set -euxo pipefail

sudo apt update
sudo apt install -y   build-essential cmake ninja-build git pkg-config   qt6-base-dev qt6-base-dev-tools qt6-multimedia-dev qmake6   libopencv-dev   libhdf5-dev   libpugixml-dev   libdcmtk-dev   libismrmrd-dev   zlib1g-dev libaec-dev   libcharls-dev   libfftw3-dev   libomp-dev
```

Sanity checks:

```bash
set -euxo pipefail
echo "[DBG] pkg-config sanity:"
pkg-config --modversion opencv4  || true
pkg-config --modversion hdf5     || true
pkg-config --modversion pugixml  || true
pkg-config --modversion dcmtk    || true
pkg-config --modversion ismrmrd  || true
```

**Notes**
- On Jetson/Ubuntu, `libismrmrd-dev` often **does not ship an `ismrmrd.pc`**. The project already supports falling back to manual include/lib linking.
- HDF5 HL/C++ `pkg-config` entries (`hdf5_hl`, `hdf5_cpp`, etc.) are often missing. The project already supports manual include/lib for those too.

---

## 2) Clone the repo

```bash
set -euxo pipefail
cd ~
git clone https://github.com/agustinTortolero/glimpse_mri.git
cd glimpse_mri

# Verify expected layout
test -d gui && test -d engine && test -d dicom_io_lib
```

---

## 3) Build `dicom_io_lib` (DICOM support)

```bash
set -euxo pipefail
cd ~/glimpse_mri

mkdir -p gui/release

cd dicom_io_lib
qmake6 dicom_io_lib.pro CONFIG+=release
make clean
make -j"$(nproc)"

# Copy output to GUI release folder
cp -v ./libdicom_io_lib.so ../gui/release/
ls -la ../gui/release | grep dicom
```

---

## 4) Build `mri_engine` (CPU + CUDA)

The engine supports a CUDA build using a qmake config flag (example: `CONFIG+=with_cuda`).

```bash
set -euxo pipefail
export PATH="/usr/local/cuda/bin:${PATH}"

cd ~/glimpse_mri/engine
qmake6 mri_engine.pro CONFIG+=release CONFIG+=with_cuda
make clean
make -j"$(nproc)"

# Engine should land in gui/release (either via DESTDIR in .pro or your build copy step)
ls -la ../gui/release | grep mri_engine
```

Confirm CUDA libraries are linked:

```bash
set -euxo pipefail
ldd ../gui/release/libmri_engine.so | egrep "cudart|cufft|cublas" || true
ldd ../gui/release/libmri_engine.so | grep -i "not found" && echo "[ERR] missing libs" && exit 2 || true
```

---

## 5) Build the Qt GUI

```bash
set -euxo pipefail
cd ~/glimpse_mri/gui
qmake6 glimpseMRI.pro CONFIG+=release
make clean
make -j"$(nproc)"
```

Expected output:
- GUI executable: `~/glimpse_mri/gui/glimpseMRI`
- Shared libs: `~/glimpse_mri/gui/release/libmri_engine.so*` and `libdicom_io_lib.so`

---

## 6) Run + logs

### Run on the Jetson desktop (recommended)
```bash
set -euxo pipefail
cd ~/glimpse_mri/gui
export LD_LIBRARY_PATH="$PWD/release:$PWD:$LD_LIBRARY_PATH"
./glimpseMRI
```

### Run headless (no display) for smoke testing
```bash
set -euxo pipefail
cd ~/glimpse_mri/gui
export LD_LIBRARY_PATH="$PWD/release:$PWD:$LD_LIBRARY_PATH"
QT_QPA_PLATFORM=offscreen ./glimpseMRI
```

### Logs
The app writes to:

- `~/Documents/GlimpseMRI/logs/app.log`

Useful greps:

```bash
LOG="$HOME/Documents/GlimpseMRI/logs/app.log"

# App startup / engine init
grep -nE "\[DBG\]\[Main\]|Engine|ENGINE" "$LOG" | tail -n 120

# Reconstruction runtime (ms)
grep -nE "\[CTRL\]\[LOAD\].*LIB recon ok=.*ms=" "$LOG" | tail -n 20

# CUDA status
grep -nEi "CUDA|cuda_compiled|has_cuda|backend not compiled|device|cufft|cudart|cublas" "$LOG" | tail -n 120
```

---

## Troubleshooting

### “could not connect to display” / “no Qt platform plugin could be initialized”
You’re running from SSH without an X session. Fix by:
- running from the Jetson’s local desktop terminal, **or**
- using `ssh -X` / `ssh -Y`, **or**
- using `QT_QPA_PLATFORM=offscreen` for headless.

### Verify which engine `.so` is being loaded
If CUDA is linked but the log still says “backend not compiled”, confirm the runtime loader path:

```bash
cd ~/glimpse_mri/gui
LD_DEBUG=libs ./glimpseMRI 2>&1 | grep -i libmri_engine | head -n 80
```

---


