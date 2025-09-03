// model/mri_engine.hpp  (DROP-IN REPLACEMENT)
#pragma once
#include <vector>
#include <complex>
#include <string>

namespace mri {

struct KSpace {
    int coils = 0;
    int ny = 0;
    int nx = 0;
    // Layout: [C, ny, nx]
    std::vector<std::complex<float>> host;
};

// Initialize CUDA and print device info (safe to call multiple times)
void init_cuda_and_versions(int device = 0);

// Compute IFFT (2D) per coil + Root-Sum-of-Squares on GPU.
// Returns true on success. 'out_img' is ny*nx float image (magnitude).
bool ifft_rss_gpu(const KSpace& ks, std::vector<float>& out_img, std::string* dbg_log = nullptr);

} // namespace mri
