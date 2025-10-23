#pragma once
#include <vector>
#include <complex>

// CPU fallback for: C coils, image ny x nx, interleaved coils in ks_host.
bool fft_and_rss_cpu(int C, int ny, int nx,
    const std::complex<float>* ks_host,
    std::vector<float>& out_rss);
