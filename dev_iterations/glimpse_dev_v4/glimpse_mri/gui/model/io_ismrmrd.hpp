#pragma once
#include <string>
#include <vector>
#include <complex>
#include <ismrmrd/dataset.h>
#include <ismrmrd/xml.h>

// Reuse KSpace and signatures consistent with your fastMRI loader
// (KSpace is declared in io_fastmri.hpp)
#include "io_fastmri.hpp"

namespace mri {

/**
 * Load ISMRMRD HDF5.
 * - If a pre-reconstructed image is found (e.g., /dataset/image_0, /dataset/images_0/data),
 *   fill *preRecon (float, row-major), with preNy/preNx set, and return true.
 * - If a dense k-space dataset is found that looks like [C,ny,nx,2] or [S,C,ny,nx,2],
 *   fill ks.host with std::complex<float> and set ks.coils/ny/nx, and return true.
 * - If both are present, both are returned (caller decides which to use).
 */
bool load_ismrmrd_any(const std::string& path,
                      KSpace& ks,
                      std::vector<float>* preRecon,
                      int* preNy,
                      int* preNx,
                      std::string* dbg);

} // namespace mri
