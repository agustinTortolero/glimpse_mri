// mri_engine.cu
#include "mri_engine.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <sstream>
#include <algorithm>

namespace mri {

// ---------- helpers ----------
#define CUDA_CHECK(expr) do { \
cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[CUDA][ERR] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        goto fail; \
} \
} while(0)

    static inline dim3 grid1D(int n, int tpb=256) {
        return dim3((n + tpb - 1) / tpb);
    }


// In mri_engine.cu (inside namespace mri)
void init_cuda_and_versions(int device /*=0*/) {
    // Debug: show CUDA runtime/driver and device info
    int drv = 0, rt = 0, count = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    fprintf(stderr, "[CUDA][DBG] runtime=%d driver=%d\n", rt, drv);

    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) {
        fprintf(stderr, "[CUDA][ERR] cudaGetDeviceCount: %s\n", cudaGetErrorString(e));
        return;
    }
    fprintf(stderr, "[CUDA][DBG] device_count=%d\n", count);
    if (count == 0) {
        fprintf(stderr, "[CUDA][ERR] no CUDA devices\n");
        return;
    }
    if (device < 0 || device >= count) {
        fprintf(stderr, "[CUDA][WRN] requested device %d out of range, using 0\n", device);
        device = 0;
    }

    e = cudaSetDevice(device);
    if (e != cudaSuccess) {
        fprintf(stderr, "[CUDA][ERR] cudaSetDevice(%d): %s\n", device, cudaGetErrorString(e));
        return;
    }

    cudaDeviceProp prop{};
    e = cudaGetDeviceProperties(&prop, device);
    if (e == cudaSuccess) {
        fprintf(stderr,
            "[CUDA][DBG] using device %d: \"%s\", CC %d.%d, globalMem %.1f MB, smCount %d\n",
            device, prop.name, prop.major, prop.minor,
            prop.totalGlobalMem / (1024.0 * 1024.0), prop.multiProcessorCount);
    } else {
        fprintf(stderr, "[CUDA][ERR] cudaGetDeviceProperties: %s\n", cudaGetErrorString(e));
    }

    // warm-up to init context
    cudaFree(0);
}


// scale complex array in-place: z *= s
__global__ void k_scale_complex(float2* a, int N, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float2 v = a[i];
        v.x *= s; v.y *= s;
        a[i] = v;
    }
}

// RSS over coils: out[y,x] = sqrt( sum_c |img_c[y,x]|^2 )
__global__ void k_rss(const float2* imgs, int C, int H, int W, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // linear over H*W
    const int HW = H * W;
    if (i < HW) {
        float acc = 0.f;
        // imgs layout: [C][H][W] contiguous
        for (int c = 0; c < C; ++c) {
            const float2 z = imgs[c * HW + i];
            acc += z.x * z.x + z.y * z.y;
        }
        out[i] = sqrtf(acc);
    }
}

// fftshift on magnitude (float) image
__global__ void k_fftshift_f32(const float* in, int H, int W, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int HW = H * W;
    if (idx < HW) {
        int y = idx / W;
        int x = idx - y * W;
        int yy = (y + H/2) % H;
        int xx = (x + W/2) % W;
        out[y * W + x] = in[yy * W + xx];
    }
}

// center crop square s×s from (H×W) into out (s×s)
__global__ void k_crop_center_square(const float* in, int H, int W, float* out, int s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int SS = s * s;
    if (idx < SS) {
        int y = idx / s;
        int x = idx - y * s;
        int offY = (H - s) / 2;
        int offX = (W - s) / 2;
        out[idx] = in[(y + offY) * W + (x + offX)];
    }
}

bool ifft_rss_gpu(const mri::KSpace& ks,
                  std::vector<float>& out,
                  int& outH, int& outW,
                  std::string* dbg)
{
    std::ostringstream os;
    const int C  = ks.coils;
    const int H  = ks.ny;
    const int W  = ks.nx;
    const int HW = H * W;
    const size_t Ncx = static_cast<size_t>(C) * HW;

    os << "[DBG][Recon] IFFT RSS GPU start (per-coil plan2d). C=" << C
       << " H=" << H << " W=" << W << "\n";

    if (C <= 0 || H <= 0 || W <= 0 || ks.host.size() != Ncx) {
        os << "[ERR][Recon] invalid dims or buffer. host=" << ks.host.size()
        << " expect=" << Ncx << "\n";
        if (dbg) *dbg += os.str();
        return false;
    }

    // --- device buffers ---
    float2* d_k = nullptr;         // k-space / image buffer for all coils (in-place IFFT)
    float*  d_mag = nullptr;       // magnitude H×W
    float*  d_shift = nullptr;     // shifted H×W
    float*  d_crop = nullptr;      // cropped s×s (if cropping)
    CUDA_CHECK(cudaMalloc(&d_k,    Ncx * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_mag,  HW  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_shift,HW  * sizeof(float)));

    // Copy host complex -> device float2
    static_assert(sizeof(std::complex<float>) == sizeof(float2),
                  "std::complex<float> must be layout-compatible with float2");
    CUDA_CHECK(cudaMemcpy(d_k, ks.host.data(),
                          Ncx * sizeof(float2), cudaMemcpyHostToDevice));

    // --- cuFFT plan (2D, reused per coil) ---
    cufftHandle plan = 0;
    if (cufftPlan2d(&plan, H, W, CUFFT_C2C) != CUFFT_SUCCESS) {
        os << "[ERR][Recon] cuFFT plan2d failed\n";
        goto fail;
    }
    os << "[DBG][Recon] cuFFT plan2d created for (" << H << "," << W << ")\n";

    // --- IFFT per coil (in-place) ---
    for (int c = 0; c < C; ++c) {
        float2* plane = d_k + static_cast<size_t>(c) * HW;
        if (cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(plane),
                         reinterpret_cast<cufftComplex*>(plane),
                         CUFFT_INVERSE) != CUFFT_SUCCESS) {
            os << "[ERR][Recon] cufftExecC2C failed at coil " << c << "\n";
            goto fail;
        }
    }
    os << "[DBG][Recon] All coils IFFT done.\n";

    // --- scale (cuFFT doesn't scale inverse) ---
    {
        const float s = 1.0f / static_cast<float>(HW);
        k_scale_complex<<<grid1D(Ncx), 256>>>(d_k, static_cast<int>(Ncx), s);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        os << "[DBG][Recon] Scale 1/(" << H << "*" << W << ") applied.\n";
    }

    // --- RSS to magnitude (H×W floats) ---
    {
        k_rss<<<grid1D(HW), 256>>>(d_k, C, H, W, d_mag);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        os << "[DBG][Recon] RSS computed.\n";
    }

    // --- fftshift on magnitude ---
    {
        k_fftshift_f32<<<grid1D(HW), 256>>>(d_mag, H, W, d_shift);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        os << "[DBG][Recon] img fftshift applied: offY=" << (H/2) << " offX=" << (W/2) << "\n";
    }

    // --- center-crop to square (min(H,W)×min(H,W)) ---
    {
        const int s = std::min(H, W);
        if (s > 0 && (H != s || W != s)) {
            CUDA_CHECK(cudaMalloc(&d_crop, s * s * sizeof(float)));
            k_crop_center_square<<<grid1D(s*s), 256>>>(d_shift, H, W, d_crop, s);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            outH = s; outW = s;
            out.resize(static_cast<size_t>(s) * s);
            CUDA_CHECK(cudaMemcpy(out.data(), d_crop, out.size()*sizeof(float), cudaMemcpyDeviceToHost));
            os << "[DBG][Recon] IFFT RSS GPU done.\n";
            os << "[DBG][Recon] Done (plan2d/coil-loop). -> center-crop " << s << "x" << s << "\n";
        } else {
            outH = H; outW = W;
            out.resize(static_cast<size_t>(HW));
            CUDA_CHECK(cudaMemcpy(out.data(), d_shift, out.size()*sizeof(float), cudaMemcpyDeviceToHost));
            os << "[DBG][Recon] IFFT RSS GPU done (full size " << H << "x" << W << ").\n";
        }
    }

    // cleanup
    cufftDestroy(plan);
    cudaFree(d_k); cudaFree(d_mag); cudaFree(d_shift);
    if (d_crop) cudaFree(d_crop);
    if (dbg) *dbg += os.str();
    return true;

fail:
    if (plan) cufftDestroy(plan);
    if (d_k) cudaFree(d_k);
    if (d_mag) cudaFree(d_mag);
    if (d_shift) cudaFree(d_shift);
    if (d_crop) cudaFree(d_crop);
    out.clear(); outH = outW = 0;
    os << "[Recon][ERR] ifft_rss_gpu failed\n";
    if (dbg) *dbg += os.str();
    return false;
}

} // namespace mri
