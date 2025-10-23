#pragma once
#include "common.hpp"
bool load_ismrmrd_slice(const std::string& path, int slice_idx, int step2, KsGrid& ks);
