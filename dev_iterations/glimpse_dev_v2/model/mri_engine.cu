// model/mri_engine.cu
#include "mri_engine.hpp"
#include "../src/common.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { \
cudaError_t err__ = (x); \
    if (err__ != cudaSuccess) { \
        std::cerr << "[ERR][CUDA] " #x " -> " << cudaGetErrorString(err__) << "\n"; \
        return false; \
} \
} while(0)
#endif

    namespace mri {

    // ---------------- kernels (top-level, no lambdas) ----------------

    __device__ __forceinline__ size_t idx3(int c, int y, int x, int C, int H, int W) {
        return (size_t)c * H * W + (size_t)y * W + (size_t)x;
    }

    __global__ void k_ifftshift2d_off(const cufftComplex* __restrict__ in,
                                      cufftComplex* __restrict__ out,
                                      int C, int H, int W, int offY, int offX)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int c = blockIdx.z;
        if (x >= W || y >= H || c >= C) return;

        const int sx = (x + offX) % W;
        const int sy = (y + offY) % H;

        out[idx3(c, y, x, C, H, W)] = in[idx3(c, sy, sx, C, H, W)];
    }

    __global__ void k_scale(cufftComplex* data, int n, float s)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            data[i].x *= s;
            data[i].y *= s;
        }
    }


    __global__ void k_shift2d_float_off(const float* __restrict__ in,
                                        float* __restrict__ out,
                                        int H, int W, int offY, int offX)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= W || y >= H) return;

        const int sx = (x + offX) % W;
        const int sy = (y + offY) % H;
        out[(size_t)y * W + x] = in[(size_t)sy * W + sx];
    }


    __global__ void k_rss(const cufftComplex* __restrict__ imgs,
                          float* __restrict__ out,
                          int C, int H, int W)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= W || y >= H) return;

        float acc = 0.f;
        for (int c = 0; c < C; ++c) {
            const size_t i = idx3(c, y, x, C, H, W);
            const float re = imgs[i].x;
            const float im = imgs[i].y;
            acc += re * re + im * im;
        }
        out[(size_t)y * W + x] = sqrtf(acc);
    }

    // ---------------- helper ----------------

    static void centerCropSquareCPU(const std::vector<float>& in, int H, int W,
                                    std::vector<float>& out, int& outH, int& outW)
    {
        const int S  = std::min(H, W);
        const int y0 = (H - S) / 2;
        const int x0 = (W - S) / 2;

        out.assign((size_t)S * S, 0.f);
        for (int y = 0; y < S; ++y) {
            const float* src = &in[(size_t)(y + y0) * W + x0];
            float*       dst = &out[(size_t)y * S];
            std::copy(src, src + S, dst);
        }
        outH = S; outW = S;
    }

    // ---------------- recon ----------------

    // inside namespace mri (or drop the namespace wrapper if your file already has it)
    // inside namespace mri
    // inside namespace mri (or at file scope if you don't use a namespace)
    bool ifft_rss_gpu(const KSpace& ks, std::vector<float>& out, std::string* dbg)
    {
        const int C = ks.coils;
        const int H = ks.ny;   // rows (ky)
        const int W = ks.nx;   // cols (kx)
        const size_t HW = (size_t)H * W;
        const size_t N  = (size_t)C * HW;

        std::cerr << "[DBG][Recon] IFFT RSS GPU start (per-coil plan2d). C=" << C
                  << " H=" << H << " W=" << W << "\n";

        // ---- pack host k-space to cufftComplex (coil-major, each coil plane contiguous) ----
        // If your loader already stores coil-major [C,H,W], this is a direct copy.
        std::vector<cufftComplex> h_k(N);
        for (size_t i = 0; i < N; ++i) {
            h_k[i].x = ks.host[i].real();
            h_k[i].y = ks.host[i].imag();
        }

        // ---- device buffers ----
        cufftComplex* d_img = nullptr;  // will hold coil-domain complex image after IFFT
        float* d_mag        = nullptr;  // RSS magnitude
        float* d_mag_center = nullptr;  // fftshifted magnitude for display/cropping

        CUDA_CHECK(cudaMalloc(&d_img, N * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_mag, HW * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mag_center, HW * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_img, h_k.data(), N * sizeof(cufftComplex), cudaMemcpyHostToDevice));

        // ---- IFFT per coil with a simple 2D plan (clarity > perf) ----
        cufftHandle plan2d = 0;
        if (cufftPlan2d(&plan2d, H, W, CUFFT_C2C) != CUFFT_SUCCESS) {
            std::cerr << "[ERR][Recon] cufftPlan2d failed for (" << H << "," << W << ")\n";
            cudaFree(d_img); cudaFree(d_mag); cudaFree(d_mag_center);
            return false;
        }
        std::cerr << "[DBG][Recon] cuFFT plan2d created for (" << H << "," << W << ")\n";

        for (int c = 0; c < C; ++c) {
            cufftComplex* ptr = d_img + (size_t)c * HW;
            if (cufftExecC2C(plan2d, ptr, ptr, CUFFT_INVERSE) != CUFFT_SUCCESS) {
                std::cerr << "[ERR][Recon] cufftExecC2C inverse failed at coil " << c << "\n";
                cufftDestroy(plan2d);
                cudaFree(d_img); cudaFree(d_mag); cudaFree(d_mag_center);
                return false;
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cerr << "[DBG][Recon] All coils IFFT done.\n";

        // ---- scale (cuFFT backward is unnormalized) ----
        {
            const float scale = 1.0f / float(H * W);
            const int threads = 256;
            const int blocks  = int((N + threads - 1) / threads);
            k_scale<<<blocks, threads>>>(d_img, (int)N, scale);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::cerr << "[DBG][Recon] Scale 1/(" << H << "*" << W << ") applied.\n";
        }

        // ---- RSS across coils ----
        {
            dim3 blk(16,16,1);
            dim3 grd((W + blk.x - 1) / blk.x, (H + blk.y - 1) / blk.y, 1);
            k_rss<<<grd, blk>>>(d_img, d_mag, C, H, W);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::cerr << "[DBG][Recon] RSS computed.\n";
        }

        // ---- fftshift in IMAGE domain to center DC before cropping ----
        {
            const int offY = H / 2, offX = W / 2;
            dim3 blk(16,16,1);
            dim3 grd((W + blk.x - 1) / blk.x, (H + blk.y - 1) / blk.y, 1);
            k_shift2d_float_off<<<grd, blk>>>(d_mag, d_mag_center, H, W, offY, offX);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::cerr << "[DBG][Recon] img fftshift applied: offY=" << offY
                      << " offX=" << offX << "\n";
        }

        // ---- copy back & center-crop to square ----
        std::vector<float> h_mag(HW);
        CUDA_CHECK(cudaMemcpy(h_mag.data(), d_mag_center, HW * sizeof(float), cudaMemcpyDeviceToHost));

        auto centerCropSquareCPU = [](const std::vector<float>& in, int H, int W,
                                      std::vector<float>& out, int& outH, int& outW)
        {
            const int S  = std::min(H, W);
            const int y0 = (H - S) / 2;
            const int x0 = (W - S) / 2;
            out.assign((size_t)S * S, 0.f);
            for (int y = 0; y < S; ++y) {
                const float* src = &in[(size_t)(y + y0) * W + x0];
                float*       dst = &out[(size_t)y * S];
                std::copy(src, src + S, dst);
            }
            outH = S; outW = S;
        };

        int outH = 0, outW = 0;
        centerCropSquareCPU(h_mag, H, W, out, outH, outW);

        if (dbg) {
            *dbg += "[DBG][Recon] Done (plan2d/coil-loop). -> center-crop "
                    + std::to_string(outH) + "x" + std::to_string(outW) + "\n";
        }

        // ---- cleanup ----
        cufftDestroy(plan2d);
        cudaFree(d_img);
        cudaFree(d_mag);
        cudaFree(d_mag_center);

        std::cerr << "[DBG][Recon] IFFT RSS GPU done.\n";
        return true;
    }


    } // namespace mri

