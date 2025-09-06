#pragma once
// mri_engine_api.h — C ABI for QLibrary::resolve()

#include "mri_engine_export.hpp"

#ifdef __cplusplus
extern "C" {
#endif

// Initialize CUDA (idempotent). Returns 1 on success, 0 on failure.
MRI_ENGINE_API int mri_engine_init(void);

// Simple version string owned by the DLL (do not free).
MRI_ENGINE_API const char* mri_engine_version(void);

// Shutdown hooks if you need them later (currently a no-op).
MRI_ENGINE_API void mri_engine_shutdown(void);

// Perform IFFT+RSS reconstruction from interleaved complex k-space on host.
// Layout: kspace_ci has length = 2 * coils * ny * nx, with (real, imag) pairs.
// On success, returns 1 and writes outH/outW and 'out' image (size outH*outW).
MRI_ENGINE_API int mri_ifft_rss_interleaved(const float* kspace_ci,
                                            int coils, int ny, int nx,
                                            float* out,
                                            int* outH, int* outW,
                                            char* logbuf, int logbuf_len);

#ifdef __cplusplus
} // extern "C"
#endif
