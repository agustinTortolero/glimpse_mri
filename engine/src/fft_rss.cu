// src/fft_rss.cu — CUDA backend for inverse FFT + Root-Sum-of-Squares (RSS)
// Clarity-first, with debug prints. Mirrors the CPU path's *unnormalized* iFFT.
//
// Signature used by engine.cpp:
//   bool fft_and_rss_cuda(int C, int ny, int nx,
//                         const std::complex<float>* ks,
//                         std::vector<float>& out_rss);

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <complex>


//check this flag
#if defined(__CUDACC__)

#include <cuda_runtime.h>
#include <cufft.h>

using cfloat = cufftComplex;

// --------- Debug helpers ----------------------------------------------------
static inline void dbg(const char* msg) {
    if (msg && *msg) std::cerr << msg << "\n";
}
static inline void dbg_i(const char* tag, long long v) {
    std::cerr << "[DBG][CUDA] " << tag << v << "\n";
}

// --------- Error checking ---------------------------------------------------
#define CHECK_CUDA(call) do {                                         \
      cudaError_t _e = (call);                                          \
      if (_e != cudaSuccess) {                                          \
          std::cerr << "[ERR][CUDA] " #call " -> "                      \
                    << cudaGetErrorString(_e) << "\n";                  \
          return false;                                                 \
      }                                                                 \
  } while(0)

static const char* cufft_err_str(cufftResult r) {
    switch (r) {
    case CUFFT_SUCCESS: return "SUCCESS";
    case CUFFT_INVALID_PLAN: return "INVALID_PLAN";
    case CUFFT_ALLOC_FAILED: return "ALLOC_FAILED";
    case CUFFT_INVALID_TYPE: return "INVALID_TYPE";
    case CUFFT_INVALID_VALUE: return "INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED: return "EXEC_FAILED";
    case CUFFT_SETUP_FAILED: return "SETUP_FAILED";
    case CUFFT_INVALID_SIZE: return "INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA: return "UNALIGNED_DATA";
    default: return "UNKNOWN";
    }
}
#define CHECK_CUFFT(call) do {                                        \
      cufftResult _r = (call);                                          \
      if (_r != CUFFT_SUCCESS) {                                        \
          std::cerr << "[ERR][CUFFT] " #call " -> "                     \
                    << cufft_err_str(_r) << " (" << int(_r) << ")\n";   \
          return false;                                                 \
      }                                                                 \
  } while(0)

// --------- Kernels ---------------------------------------------------------
// imgC layout: C contiguous images, each N=ny*nx elements
// Accumulates sum(|img_c|^2) over coils into out_accum[N]
__global__ void rss_accumulate(const cfloat* __restrict__ imgC,
    float* __restrict__ out_accum,
    int C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float s = 0.f;
    const int stride = N; // distance between coils
#pragma unroll 1
    for (int c = 0; c < C; ++c) {
        cfloat v = imgC[c * stride + i];
        s += v.x * v.x + v.y * v.y;
    }
    out_accum[i] = s;
}

__global__ void rss_sqrt_inplace(float* __restrict__ buf, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    buf[i] = sqrtf(buf[i]);
}

// --------- Implementation ---------------------------------------------------
bool fft_and_rss_cuda(int C, int ny, int nx,
    const std::complex<float>* ks_host,
    std::vector<float>& out_rss)
{
    // --- sanity -------------------------------------------------------------
    if (C <= 0 || ny <= 0 || nx <= 0 || ks_host == nullptr) {
        std::cerr << "[ERR][CUDA] invalid args (C=" << C
            << " ny=" << ny << " nx=" << nx << ")\n";
        return false;
    }

    const int    N = ny * nx;
    const size_t plane = static_cast<size_t>(N);
    const size_t total = static_cast<size_t>(C) * plane;
    out_rss.assign(plane, 0.f);

    std::cerr << "[DBG][CUDA] fft_and_rss_cuda start  C=" << C
        << " ny=" << ny << " nx=" << nx << "  N=" << N << "\n";

    // --- device buffers -----------------------------------------------------
    cfloat* d_kspace = nullptr;
    cfloat* d_img = nullptr;
    float* d_rss = nullptr;

    CHECK_CUDA(cudaMalloc(&d_kspace, total * sizeof(cfloat)));
    CHECK_CUDA(cudaMalloc(&d_img, total * sizeof(cfloat)));
    CHECK_CUDA(cudaMalloc(&d_rss, plane * sizeof(float)));

    // Convert host k-space (std::complex<float>) -> host cufftComplex buffer
    // (clarity > perf; avoids aliasing assumptions).
    std::vector<cfloat> h_kspace(total);
    for (size_t i = 0; i < total; ++i) {
        h_kspace[i].x = ks_host[i].real();
        h_kspace[i].y = ks_host[i].imag();
    }

    // H2D
    CHECK_CUDA(cudaMemcpy(d_kspace, h_kspace.data(),
        total * sizeof(cfloat), cudaMemcpyHostToDevice));
    std::cerr << "[DBG][CUDA] H2D bytes=" << (total * sizeof(cfloat)) << "\n";

    // --- cuFFT plan: 2D inverse, batched C ---------------------------------
    cufftHandle plan;
    int n[2] = { ny, nx };

    // tightly packed images
    int istride = 1, ostride = 1;
    int idist = N, odist = N;
    int inembed[2] = { ny, nx };
    int onembed[2] = { ny, nx };

    CHECK_CUFFT(cufftPlanMany(&plan,
        /*rank*/2, n,
        /*inembed*/  inembed, istride, idist,
        /*onembed*/  onembed, ostride, odist,
        CUFFT_C2C, /*batch=*/ C));
    std::cerr << "[DBG][CUFFT] plan created (batch=" << C
        << ", N=" << N << ")\n";

    // Execute inverse FFT (k-space -> image), unnormalized
    CHECK_CUFFT(cufftExecC2C(plan, d_kspace, d_img, CUFFT_INVERSE));
    std::cerr << "[DBG][CUFFT] exec done\n";

    // --- RSS kernels --------------------------------------------------------
    // 1) accumulate sum(|coil|^2) over coils
    {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        rss_accumulate << <blocks, threads >> > (d_img, d_rss, C, N);
        CHECK_CUDA(cudaGetLastError());
        std::cerr << "[DBG][CUDA] rss_accumulate grid=" << blocks
            << " block=" << threads << "\n";
    }

    // 2) sqrt
    {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        rss_sqrt_inplace << <blocks, threads >> > (d_rss, N);
        CHECK_CUDA(cudaGetLastError());
        std::cerr << "[DBG][CUDA] rss_sqrt_inplace grid=" << blocks
            << " block=" << threads << "\n";
    }

    // --- D2H ---------------------------------------------------------------
    CHECK_CUDA(cudaMemcpy(out_rss.data(), d_rss,
        plane * sizeof(float), cudaMemcpyDeviceToHost));

    // --- cleanup ------------------------------------------------------------
    cufftDestroy(plan);
    cudaFree(d_kspace);
    cudaFree(d_img);
    cudaFree(d_rss);

    std::cerr << "[DBG][CUDA] fft_and_rss_cuda done\n";
    return true;
}

#else  // !__CUDACC__

  // --------- Non-CUDA build stub ---------------------------------------------
bool fft_and_rss_cuda(int C, int ny, int nx,
    const std::complex<float>* ks_host,
    std::vector<float>& out_rss)
{
    (void)C; (void)ny; (void)nx; (void)ks_host; (void)out_rss;
    std::cerr << "[DBG][CUDA] fft_and_rss_cuda: CUDA not enabled at build.\n";
    return false;
}

#endif // __CUDACC__
