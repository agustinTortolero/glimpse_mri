mri__engine_lib — CUDA IFFT+RSS DLL (QLibrary-friendly)
======================================================

This package builds a Windows DLL that exposes a **C ABI** you can load with Qt's `QLibrary`.
Internally it calls your `mri::init_cuda_and_versions()` and `mri::ifft_rss_gpu(...)` from `mri_engine.cu/.hpp`.

Files
-----
- `mri__engine_lib.pro` — qmake project for a **shared library**.
- `include/mri_engine_export.hpp` — `__declspec(dllexport/dllimport)` macro.
- `include/mri_engine_api.h` — C-ABI header (what your Qt app should include).
- `include/mri_engine.hpp` — C++ header consumed by the CUDA implementation.
- `src/api_exports.cpp` — thin C wrappers with **verbose debug prints**.
- `src/mri_engine.cu` — your CUDA implementation (as provided).
- `samples/qt_loader_example.cpp` — tiny example of loading with `QLibrary`.

Build (Qt Creator / qmake, MSVC, CUDA 12.x)
-------------------------------------------
1. Open `mri__engine_lib.pro` in Qt Creator (MSVC 64-bit kit).
2. If needed, set an environment variable `CUDA_PATH` to your install, e.g.
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`
3. Optionally set `VCPKG_ROOT` if you want extra headers available.
4. Build. The DLL will be placed in `bin\` (Debug/Release).

Notes
-----
- Default GPU arch is `sm_89` (Ada / RTX 40xx). For Jetson Orin change to `CUDA_ARCH = 87`.
- The exported function `mri_ifft_rss_interleaved` expects interleaved complex k-space
  `float` pairs of size `2*C*H*W`. It writes an `H×W` float image.
- All functions print **[DBG]** / **[ERR]** messages to `stderr` to ease debugging.

Have fun!
