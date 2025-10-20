// model/io.hpp
#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace io {

// High-level probe classification (kept for UI; now extension/magic only)
enum class Flavor {
    FastMRI,                // (unused now; DLL will handle)
    ISMRMRD_Cartesian,      // (unused now; DLL will handle)
    ISMRMRD_NonCartesian,   // (unused now; DLL will handle)
    ISMRMRD_Unknown,        // (unused now; DLL will handle)
    HDF5_Unknown,           // any .h5/.hdf5 -> let DLL figure it out
    DICOM,
    NotHDF5
};

struct ProbeResult {
    Flavor      flavor = Flavor::NotHDF5;
    std::string reason;
    std::string trajectory;     // optional UI hint
    bool has_xml          = false;
    bool has_acq          = false;
    bool has_kspace       = false;
    bool has_embedded_img = false;
};

// Minimal, file-signature-based probe (no HDF5 dependencies)
ProbeResult probe(const std::string& path, std::string* dbg = nullptr);

// -------- DICOM + simple writers (unchanged APIs) --------
bool read_dicom_gray8(const std::string& path, cv::Mat& out8, std::string* why);

bool read_dicom_frames_gray8(const std::string& path,
                             std::vector<cv::Mat>& out,
                             std::string* why);

bool read_dicom_frames_u16(const std::string& path,
                           std::vector<cv::Mat>& out16,
                           std::string* why);

bool write_png(const std::string& path_utf8, const cv::Mat& img8, std::string* why);
bool write_dicom_sc_gray8(const std::string& path, const cv::Mat& img8, std::string* why);

// --- Basic DICOM/ISMRMRD metadata for UI ---
struct DicomMeta {
    std::string manufacturer;        // (0008,0070) or ismrmrdHeader/acquisitionSystemInformation/systemVendor
    std::string modelName;           // (0008,1090) or .../systemModel
    std::string softwareVersions;    // (0018,1020)
    std::string institutionName;     // (0008,0080) or .../institutionName
    std::string seriesDescription;   // (0008,103E)
    std::string patientName;         // (0010,0010)
    std::string patientID;           // (0010,0020)
    std::string studyDate;           // (0008,0020)
    std::string studyTime;           // (0008,0030)
    std::string B0T;                 // MagneticFieldStrength text (0018,0087) or .../systemFieldStrength_T

    // New: sequence timing (ms)
    std::string tr_ms;               // DICOM (0018,0080) RepetitionTime OR ISMRMRD sequenceParameters/TR
    std::string te_ms;               // DICOM (0018,0081) EchoTime      OR ISMRMRD sequenceParameters/TE
    std::string ti_ms;               // DICOM (0018,0082) InversionTime OR ISMRMRD sequenceParameters/TI
};

// Call once early (e.g., AppController ctor) before reading DICOMs
void dcmtk_global_init();
void dcmtk_global_shutdown();

// Read basic DICOM (file/series) metadata
bool read_dicom_basic_meta(const std::string& path, DicomMeta& out, std::string* why);

// Read ISMRMRD XML inside HDF5 (fastMRI/ISMRMRD)
bool read_hdf5_ismrmrd_meta(const std::string& path, DicomMeta& out, std::string* why);



} // namespace io
