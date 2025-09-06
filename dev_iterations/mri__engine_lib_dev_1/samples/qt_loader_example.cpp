// samples/qt_loader_example.cpp
// Minimal demo showing how to load the DLL via QLibrary and call the C API.
#include <QLibrary>
#include <QDebug>
#include <vector>
#include <cmath>

#include "../include/mri_engine_api.h" // for typedefs — or you can redeclare.

// When loading dynamically, you usually typedef the signatures:
typedef int (*PFN_mri_engine_init)(void);
typedef const char* (*PFN_mri_engine_version)(void);
typedef void (*PFN_mri_engine_shutdown)(void);
typedef int (*PFN_mri_ifft_rss_interleaved)(const float*, int, int, int, float*, int*, int*, char*, int);

int main() {
    QLibrary lib("mri__engine_lib"); // Windows will pick mri__engine_lib.dll
    if (!lib.load()) {
        qCritical() << "Failed to load mri__engine_lib:" << lib.errorString();
        return 1;
    }
    auto p_init = (PFN_mri_engine_init)lib.resolve("mri_engine_init");
    auto p_ver  = (PFN_mri_engine_version)lib.resolve("mri_engine_version");
    auto p_shut = (PFN_mri_engine_shutdown)lib.resolve("mri_engine_shutdown");
    auto p_rec  = (PFN_mri_ifft_rss_interleaved)lib.resolve("mri_ifft_rss_interleaved");

    if (!p_init || !p_ver || !p_shut || !p_rec) {
        qCritical() << "Resolve failed!"
                    << !!p_init << !!p_ver << !!p_shut << !!p_rec;
        return 2;
    }

    qDebug() << "DLL version:" << p_ver();
    if (!p_init()) {
        qCritical() << "Init failed";
        return 3;
    }

    // Build a tiny synthetic k-space (1 coil, 8x8) — just for smoke test.
    const int C=1, H=8, W=8;
    std::vector<float> kci(2*C*H*W, 0.0f);
    for (int y=0; y<H; ++y) {
        for (int x=0; x<W; ++x) {
            const int i = y*W + x;
            // Simple delta at (0,0) in image domain -> all-ones in k-space
            // We'll fill real=1.0 everywhere to keep it trivial
            kci[2*i+0] = 1.0f; // real
            kci[2*i+1] = 0.0f; // imag
        }
    }
    std::vector<float> out(H*W, 0.0f);
    int oH=0, oW=0;
    char logbuf[2048] = {0};

    const int ok = p_rec(kci.data(), C, H, W, out.data(), &oH, &oW, logbuf, sizeof(logbuf));
    qDebug() << "recon ok?" << ok << "out size" << oH << "x" << oW;
    if (logbuf[0]) qDebug() << "[log]" << logbuf;

    p_shut();
    return 0;
}
