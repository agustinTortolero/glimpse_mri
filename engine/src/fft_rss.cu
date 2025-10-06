#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <complex>

using cfloat = cufftComplex;

#define CUDA_CHK(x) do{auto e=(x); if(e!=cudaSuccess){   std::cerr<<"[DBG][CUDA] " #x " -> "<<cudaGetErrorString(e)<<"\n"; return false; }}while(0)
#define CUFFT_CHK(x) do{auto r=(x); if(r!=CUFFT_SUCCESS){   std::cerr<<"[DBG][CUFFT] " #x " -> err="<<int(r)<<"\n"; return false; }}while(0)

__global__
void rss_kernel(const cfloat* __restrict__ imgC, float* __restrict__ out,
                int C, int ny, int nx)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    size_t plane = size_t(ny) * nx;
    float s = 0.f;
    for (int c=0;c<C;++c){
        cfloat v = imgC[size_t(c)*plane + size_t(y)*nx + x];
        s += v.x*v.x + v.y*v.y;
    }
    out[size_t(y)*nx + x] = sqrtf(s);
}

bool fft_and_rss_cuda(int C, int ny, int nx,
                      const std::complex<float>* h_kspace,
                      std::vector<float>& h_rss)
{
    std::cerr << "[DBG][CUDA] fft_and_rss_cuda start C="<<C<<" ny="<<ny<<" nx="<<nx<<"\n";
    size_t plane = size_t(ny)*nx, totalC = size_t(C)*plane;

    cfloat *d_k=nullptr,*d_img=nullptr; float *d_rss=nullptr;
    CUDA_CHK(cudaMalloc(&d_k,   totalC*sizeof(cfloat)));
    CUDA_CHK(cudaMalloc(&d_img, totalC*sizeof(cfloat)));
    CUDA_CHK(cudaMalloc(&d_rss, plane *sizeof(float)));

    CUDA_CHK(cudaMemcpy(d_k, h_kspace, totalC*sizeof(cfloat), cudaMemcpyHostToDevice));
    std::cerr << "[DBG][CUDA] H2D bytes=" << (totalC*sizeof(cfloat)) << "\n";

    int n[2] = { ny, nx }; cufftHandle plan;
    CUFFT_CHK(cufftPlanMany(&plan, 2, n, n,1,plane, n,1,plane, CUFFT_C2C, C));
    std::cerr << "[DBG][CUFFT] plan created\n";

    CUFFT_CHK(cufftExecC2C(plan, d_k, d_img, CUFFT_INVERSE));
    std::cerr << "[DBG][CUFFT] exec done\n";

    dim3 blk(16,16), grd((nx+15)/16,(ny+15)/16);
    rss_kernel<<<grd,blk>>>(d_img, d_rss, C, ny, nx);
    CUDA_CHK(cudaGetLastError());
    std::cerr << "[DBG][CUDA] rss_kernel grid=("<<grd.x<<","<<grd.y<<") block=("<<blk.x<<","<<blk.y<<")\n";

    h_rss.resize(plane);
    CUDA_CHK(cudaMemcpy(h_rss.data(), d_rss, plane*sizeof(float), cudaMemcpyDeviceToHost));

    cufftDestroy(plan);
    cudaFree(d_k); cudaFree(d_img); cudaFree(d_rss);
    std::cerr << "[DBG][CUDA] fft_and_rss_cuda done\n";
    return true;
}
