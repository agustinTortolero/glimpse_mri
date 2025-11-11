#pragma once
#include "common.hpp"

// Build flag (override in your project if ISMRMRD is not available)
#ifndef ENGINE_HAS_ISMRMRD
#define ENGINE_HAS_ISMRMRD 1
#endif

// Load ONE slice -> fills KsGrid with k-space [C, ny, nx] (coil-major).
// step2: if >= 0, filter acquisitions by kspace_encoding_step_2 == step2.
// Returns true on success (and ks filled).
bool load_ismrmrd_slice(const std::string& path, int slice_idx, int step2, KsGrid& ks);
