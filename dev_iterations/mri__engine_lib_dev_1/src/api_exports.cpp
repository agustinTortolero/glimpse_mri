// src/api_exports.cpp — C ABI wrappers around C++/CUDA implementation
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <complex>

#include "mri_engine_api.h"
#include "mri_engine.hpp"   // C++ API implemented in mri_engine.cu

// Version string stored in the DLL
static const char* kVersion = "mri__engine_lib 0.1.0 (debug C-ABI)";
static bool g_initialized = false;

extern "C" {

MRI_ENGINE_API int mri_engine_init(void) {
    fprintf(stderr, "[DBG][mri__engine_lib] mri_engine_init() called\n");
    try {
        if (!g_initialized) {
            mri::init_cuda_and_versions(0);
            g_initialized = true;
        }
        return 1;
    } catch (...) {
        fprintf(stderr, "[ERR][mri__engine_lib] init failed (exception)\n");
        return 0;
    }
}

MRI_ENGINE_API const char* mri_engine_version(void) {
    fprintf(stderr, "[DBG][mri__engine_lib] mri_engine_version() -> %s\n", kVersion);
    return kVersion;
}

MRI_ENGINE_API void mri_engine_shutdown(void) {
    fprintf(stderr, "[DBG][mri__engine_lib] mri_engine_shutdown() called (noop)\n");
}

MRI_ENGINE_API int mri_ifft_rss_interleaved(const float* kspace_ci,
                                            int coils, int ny, int nx,
                                            float* out,
                                            int* outH, int* outW,
                                            char* logbuf, int logbuf_len) {
    fprintf(stderr, "[DBG][mri__engine_lib] mri_ifft_rss_interleaved(C=%d,H=%d,W=%d)\n",
            coils, ny, nx);
    if (!kspace_ci || !out || !outH || !outW) {
        fprintf(stderr, "[ERR][mri__engine_lib] null pointer argument(s)\n");
        return 0;
    }
    if (coils <= 0 || ny <= 0 || nx <= 0) {
        fprintf(stderr, "[ERR][mri__engine_lib] invalid dims\n");
        return 0;
    }

    const size_t N = static_cast<size_t>(coils) * ny * nx;
    std::vector<std::complex<float>> host;
    host.resize(N);

    // Convert from interleaved (real, imag) to std::complex<float>
    for (size_t i = 0; i < N; ++i) {
        host[i] = std::complex<float>(kspace_ci[2*i+0], kspace_ci[2*i+1]);
    }

    mri::KSpace ks;
    ks.coils = coils;
    ks.ny    = ny;
    ks.nx    = nx;
    ks.host  = std::move(host);

    std::vector<float> img;
    int H = 0, W = 0;
    std::string dbg;
    const bool ok = mri::ifft_rss_gpu(ks, img, H, W, &dbg);
    if (logbuf && logbuf_len > 0) {
        const int ncopy = (int)std::min<size_t>(dbg.size(), (size_t)(logbuf_len-1));
        std::memcpy(logbuf, dbg.data(), ncopy);
        logbuf[ncopy] = '\0';
    }
    if (!ok) {
        fprintf(stderr, "[ERR][mri__engine_lib] ifft_rss_gpu() returned false\n");
        return 0;
    }
    if (H <= 0 || W <= 0 || (size_t)(H*W) != img.size()) {
        fprintf(stderr, "[ERR][mri__engine_lib] invalid output image from core\n");
        return 0;
    }

    std::memcpy(out, img.data(), img.size()*sizeof(float));
    *outH = H;
    *outW = W;
    fprintf(stderr, "[DBG][mri__engine_lib] reconstruction OK -> H=%d W=%d\n", H, W);
    return 1;
}

} // extern "C"
