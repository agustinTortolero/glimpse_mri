#pragma once
#include "common.hpp"
struct FastMRIInfo{ int slices=0, coils=0, ny=0, nx=0; bool has_rss=false; };
bool fastmri_probe(const std::string& path, FastMRIInfo& info);
bool fastmri_load_rss_slice(const std::string& path, int slice_idx, std::vector<float>& rss, int& ny, int& nx);
bool fastmri_load_kspace_slice(const std::string& path, int slice_idx, KsGrid& ks);
