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

bool ifft_rss_gpu(const mri::KSpace& ks,
                  std::vector<float>& out,
                  int& outH, int& outW,
                  std::string* dbg = nullptr);
} // namespace mri
