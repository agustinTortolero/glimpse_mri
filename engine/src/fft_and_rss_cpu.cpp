#include "fft_and_rss_cpu.hpp"
#include <iostream>
#include <chrono>
#include <cmath>

#ifdef _MSC_VER
#define NOMINMAX
#endif

// FFTW (single-precision)
#include <fftw3.h>

static inline void dbg_head(const char* tag) { std::cerr << "[DBG][" << tag << "] "; }

bool fft_and_rss_cpu(int C, int ny, int nx,
    const std::complex<float>* ks_host,
    std::vector<float>& out_rss)
{
    using clk = std::chrono::high_resolution_clock;
    const int N = ny * nx;
    if (C <= 0 || ny <= 0 || nx <= 0 || ks_host == nullptr) {
        dbg_head("CPU"); std::cerr << "invalid args (C=" << C << " ny=" << ny << " nx=" << nx << ")\n";
        return false;
    }

    dbg_head("CPU"); std::cerr << "fft_and_rss_cpu start C=" << C
        << " ny=" << ny << " nx=" << nx
        << " (FFTW, OpenMP "
#ifdef _OPENMP
        << "ON"
#else
        << "OFF"
#endif
        << ")\n";

    out_rss.assign(N, 0.0f);

    // Allocate working buffers for FFTW (single pair reused across coils)
    fftwf_complex* in = fftwf_alloc_complex(N);
    fftwf_complex* out = fftwf_alloc_complex(N);
    if (!in || !out) {
        dbg_head("CPU"); std::cerr << "fftwf_alloc_complex failed\n";
        if (in)  fftwf_free(in);
        if (out) fftwf_free(out);
        return false;
    }

    // 2D inverse transform (backward) plan (ESTIMATE for quick plans)
    auto t_plan0 = clk::now();
    fftwf_plan plan = fftwf_plan_dft_2d(
        ny, nx, in, out, FFTW_BACKWARD, FFTW_ESTIMATE
    );
    auto t_plan1 = clk::now();
    if (!plan) {
        dbg_head("CPU"); std::cerr << "fftwf_plan_dft_2d failed\n";
        fftwf_free(in); fftwf_free(out);
        return false;
    }
    dbg_head("FFTW"); std::cerr << "plan created in "
        << std::chrono::duration<double, std::milli>(t_plan1 - t_plan0).count()
        << " ms\n";

    // Accumulator for sum of magnitudes^2
    std::vector<float> accum(N, 0.0f);

    // Process each coil
    for (int c = 0; c < C; ++c) {
        const auto t0 = clk::now();

        const std::complex<float>* src = ks_host + static_cast<size_t>(c) * N;

        // Load k-space into FFTW 'in'
        // Note: no Hanning/Hamming/windowing here — parity with GPU path.
        for (int i = 0; i < N; ++i) {
            in[i][0] = src[i].real();
            in[i][1] = src[i].imag();
        }

        // Inverse FFT (k-space -> image space per coil)
        fftwf_execute(plan);

        const auto t1 = clk::now();

        // Accumulate |coil_img|^2 into accum
        // No normalization to mirror cuFFT default (unnormalized).
        // If you ever normalize, do it consistently in both GPU & CPU.
#pragma omp parallel for if(N > 4096)
        for (int i = 0; i < N; ++i) {
            const float re = out[i][0];
            const float im = out[i][1];
            accum[i] += re * re + im * im;
        }

        const auto t2 = clk::now();
        dbg_head("CPU"); std::cerr << "coil " << c
            << " iFFT=" << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms, "
            << "accum=" << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";
    }

    // RSS = sqrt(sum |coil|^2)
#pragma omp parallel for if(N > 4096)
    for (int i = 0; i < N; ++i) {
        out_rss[i] = std::sqrt(accum[i]);
    }

    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);

    dbg_head("CPU"); std::cerr << "fft_and_rss_cpu done\n";
    return true;
}
