// src/fft_and_rss_cpu.cpp — portable CPU RSS using FFTW (single-precision)

#include "fft_and_rss_cpu.hpp"
#include "common.hpp"   // dbg_head(...)

#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>

// --- Feature flags (override from build if needed) ---------------------------
#if !defined(ENGINE_HAS_FFTW)
#define ENGINE_HAS_FFTW 1   // set to 0 to compile without FFTW (will stub)
#endif

// --- Windows macro guard (avoid MSVC C4005 when defined on the command line) -
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

// --- FFT backend selection ---------------------------------------------------
#if ENGINE_HAS_FFTW
  // FFTW (single-precision)
#include <fftw3.h>
#endif

// -----------------------------------------------------------------------------
// fft_and_rss_cpu: inverse FFT per coil + root-sum-of-squares
// Clarity-first debug prints. Unnormalized iFFT (parity with common cuFFT use).
// -----------------------------------------------------------------------------
bool fft_and_rss_cpu(int C, int ny, int nx,
    const std::complex<float>* ks_host,
    std::vector<float>& out_rss)
{
    using clk = std::chrono::high_resolution_clock;

    // --- sanity ----------------------------------------------------------------
    if (C <= 0 || ny <= 0 || nx <= 0 || !ks_host) {
        dbg_head("CPU"); std::cerr << "invalid args (C=" << C
            << " ny=" << ny << " nx=" << nx << ")\n";
        return false;
    }

#if !ENGINE_HAS_FFTW
    dbg_head("CPU"); std::cerr << "FFTW disabled at build; cannot reconstruct on CPU.\n";
    return false;
#else
    const int N = ny * nx;
    out_rss.assign((size_t)N, 0.0f);

    dbg_head("CPU"); std::cerr << "fft_and_rss_cpu start  C=" << C
        << " ny=" << ny << " nx=" << nx
        << " (FFTW, OpenMP "
#ifdef _OPENMP
        << "ON"
#else
        << "OFF"
#endif
        << ")\n";

    // --- allocate FFTW buffers -------------------------------------------------
    fftwf_complex* in = fftwf_alloc_complex((size_t)N);
    fftwf_complex* out = fftwf_alloc_complex((size_t)N);
    if (!in || !out) {
        dbg_head("CPU"); std::cerr << "fftwf_alloc_complex failed (N=" << N << ")\n";
        if (in)  fftwf_free(in);
        if (out) fftwf_free(out);
        return false;
    }

    // --- plan iFFT 2D (backward) ----------------------------------------------
    const auto t_plan0 = clk::now();
    fftwf_plan plan = fftwf_plan_dft_2d(
        ny, nx, in, out, FFTW_BACKWARD, FFTW_ESTIMATE
    );
    const auto t_plan1 = clk::now();

    if (!plan) {
        dbg_head("FFTW"); std::cerr << "plan creation failed\n";
        fftwf_free(in); fftwf_free(out);
        return false;
    }

    dbg_head("FFTW"); std::cerr << "plan created in "
        << std::chrono::duration<double, std::milli>(t_plan1 - t_plan0).count()
        << " ms\n";

    // --- RSS accumulator -------------------------------------------------------
    std::vector<float> accum((size_t)N, 0.0f);

    // --- per-coil processing ---------------------------------------------------
    for (int c = 0; c < C; ++c) {
        const auto t0 = clk::now();

        const std::complex<float>* src = ks_host + (size_t)c * (size_t)N;

        // load k-space into FFTW 'in'
        for (int i = 0; i < N; ++i) {
            in[i][0] = src[i].real();
            in[i][1] = src[i].imag();
        }

        // inverse FFT (k-space -> image space per coil)
        fftwf_execute(plan);
        const auto t1 = clk::now();

        // accumulate |coil_img|^2 (unnormalized)
        // If you later normalize, mirror the same factor in GPU path.
#pragma omp parallel for if(N > 4096)
        for (int i = 0; i < N; ++i) {
            const float re = out[i][0];
            const float im = out[i][1];
            accum[(size_t)i] += re * re + im * im;
        }
        const auto t2 = clk::now();

        dbg_head("CPU"); std::cerr << "coil " << c
            << " iFFT=" << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms, "
            << "accum=" << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";
    }

    // --- RSS = sqrt(sum |coil|^2) ---------------------------------------------
#pragma omp parallel for if(N > 4096)
    for (int i = 0; i < N; ++i) {
        out_rss[(size_t)i] = std::sqrt(accum[(size_t)i]);
    }

    // --- cleanup ---------------------------------------------------------------
    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);

    dbg_head("CPU"); std::cerr << "fft_and_rss_cpu done\n";
    return true;
#endif // ENGINE_HAS_FFTW
}
