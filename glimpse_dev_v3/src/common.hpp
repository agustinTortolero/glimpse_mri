#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <complex>
#include <cstdio>
#include <iostream>

#define CUDA_CHECK(expr) do {                                    \
  cudaError_t _err = (expr);                                     \
  if (_err != cudaSuccess) {                                     \
    throw std::runtime_error(std::string("CUDA error: ") +       \
      cudaGetErrorString(_err) + " at " + __FILE__ + ":" +       \
      std::to_string(__LINE__));                                 \
  }                                                              \
} while(0)

#define CUFFT_CHECK(expr) do {                                   \
  cufftResult _res = (expr);                                     \
  if (_res != CUFFT_SUCCESS) {                                   \
    throw std::runtime_error(std::string("cuFFT error code ") +  \
      std::to_string(_res) + " at " + __FILE__ + ":" +           \
      std::to_string(__LINE__));                                 \
  }                                                              \
} while(0)

namespace mri {
inline std::string device_name(int dev) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  return std::string(prop.name ? prop.name : "unknown");
}
inline int sm_count(int dev) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  return prop.multiProcessorCount;
}
inline size_t device_mem_bytes(int dev) {
  CUDA_CHECK(cudaSetDevice(dev));
  size_t free_b=0, total_b=0;
  CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  return total_b;
}
} // namespace mri