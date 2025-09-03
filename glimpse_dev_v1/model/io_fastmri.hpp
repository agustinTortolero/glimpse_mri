#pragma once
#include <string>
#include <vector>
#include "mri_engine.hpp"   // for mri::KSpace

namespace mri {

/**
 * Robust fastMRI loader.
 * - Fills KSpace (coils, ny, nx, interleaved complex).
 * - If "reconstruction_rss" exists, reads slice 0 into preRecon and reports its size via preNy/preNx.
 * - Returns true if kspace parsed OR preRecon found (so the UI can still show something).
 */
bool load_fastmri_kspace(const std::string& path,
                         KSpace& ks,
                         std::vector<float>* preRecon = nullptr,
                         int* preNy = nullptr,
                         int* preNx = nullptr,
                         std::string* dbg = nullptr);

} // namespace mri
