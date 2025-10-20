// model/io.hpp
#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "mri_engine.hpp"  // for mri::KSpace

namespace io {

// -------- File classification --------
enum class Flavor {
    FastMRI,
    ISMRMRD_Cartesian,
    ISMRMRD_NonCartesian,
    ISMRMRD_Unknown,
    HDF5_Unknown,
    DICOM,
    NotHDF5
};

struct ProbeResult {
    Flavor flavor = Flavor::HDF5_Unknown;
    std::string trajectory;         // "cartesian", "radial", ...
    bool has_xml = false;
    bool has_acq = false;
    bool has_kspace = false;
    bool has_embedded_img = false;
    std::string reason;
};

// Probe a path and classify it (HDF5 fastMRI / ISMRMRD / DICOM / not HDF5)
ProbeResult probe(const std::string& path, std::string* dbg = nullptr);

// -------- Reading --------

// Unified HDF5 loader. Routes to fastMRI or ISMRMRD under the hood.
// Returns true if either k-space OR an embedded image is obtained.
// - If k-space succeeds, ks.{coils,ny,nx,host} are set.
// - If only preRecon is found, ks is filled with 1×preH×preW just for context.
bool load_hdf5_any(const std::string& path,
                   mri::KSpace& ks,
                   std::vector<float>* preRecon,
                   int* preH,
                   int* preW,
                   std::string* dbg = nullptr);

// Simple DICOM → CV_8UC1 (first frame, min–max window if monochrome)
bool read_dicom_gray8(const std::string& path, cv::Mat& out8, std::string* why = nullptr);

// -------- Writing --------

// Save 8-bit grayscale PNG
bool write_png(const std::string& path, const cv::Mat& img8, std::string* why = nullptr);

// Save minimal Secondary Capture DICOM (8-bit MONO)
bool write_dicom_sc_gray8(const std::string& path, const cv::Mat& img8, std::string* why = nullptr);

} // namespace io
