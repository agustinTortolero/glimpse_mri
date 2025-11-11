#pragma once
#include "common.hpp"

// Simple container with basic dataset facts discovered by probe().
struct FastMRIInfo {
    int  slices = 0;
    int  coils = 0;
    int  ny = 0;
    int  nx = 0;
    bool has_rss = false;  // dataset contains "reconstruction_rss"
};

// Return true if the file looks like a fastMRI HDF5 dataset.
// Fills 'info' with dimensions and presence of RSS dataset if available.
bool fastmri_probe(const std::string& path, FastMRIInfo& info);

// Load one RSS slice (if "reconstruction_rss" exists).
// Returns true on success and fills rss (ny*nx floats), ny, nx.
bool fastmri_load_rss_slice(const std::string& path, int slice_idx,
    std::vector<float>& rss, int& ny, int& nx);

// Load one k-space slice into coil-major layout (C planes × ny × nx).
// Returns true on success and fills KsGrid: coils, ny, nx, host data.
bool fastmri_load_kspace_slice(const std::string& path, int slice_idx, KsGrid& ks);
