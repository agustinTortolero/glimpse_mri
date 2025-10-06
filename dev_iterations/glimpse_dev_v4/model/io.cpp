// model/io.cpp
#include "io.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>     // strlen, _stricmp
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// DCMTK (DICOM)
#include <dcmtk/dcmimgle/dcmimage.h>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/ofstd/ofstd.h>

// HDF5 + pugixml (for ISMRMRD XML in fastMRI/HDF5)
#include <H5Cpp.h>
#include <hdf5.h>          // for H5Dvlen_reclaim
#include <pugixml.hpp>

// ============================================================
// Debug helpers (clarity > perf)
// ============================================================
static inline void dbg_line(std::string* dbg, const std::string& s) {
    if (dbg) dbg->append(s + "\n");
    std::cerr << s << "\n";
}
static inline void dbg_line(std::string* dbg, const std::string& a, const std::string& b) {
    dbg_line(dbg, a + b);
}

namespace fs = std::filesystem;

// ============================================================
// Minimal probe: extension + magic numbers (no HDF5/ISM deps)
// ============================================================
namespace io {

static bool has_ext_ci(const std::string& path, const char* ext) {
    const size_t n = std::strlen(ext);
    if (path.size() < n) return false;
    return 0 == _stricmp(path.c_str() + path.size() - n, ext);
}

static bool is_dicom_magic(const std::string& path) {
    // DICOM: "DICM" at offset 128 (not guaranteed for all, but common)
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(128, std::ios::beg);
    char tag[4] = {0,0,0,0};
    f.read(tag, 4);
    return (f.gcount() == 4 && tag[0]=='D' && tag[1]=='I' && tag[2]=='C' && tag[3]=='M');
}

static bool is_hdf5_magic(const std::string& path) {
    // HDF5: 8-byte signature: \x89 H D F \r \n \x1A \n
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    unsigned char sig[8] = {};
    f.read(reinterpret_cast<char*>(sig), 8);
    if (f.gcount() != 8) return false;
    const unsigned char ref[8] = {0x89,'H','D','F','\r','\n',0x1A,'\n'};
    for (int i=0;i<8;++i) if (sig[i] != ref[i]) return false;
    return true;
}

ProbeResult probe(const std::string& path, std::string* dbg)
{
    ProbeResult pr{};
    dbg_line(dbg, std::string("[IO][probe] path=") + path);

    // Extension quick checks
    if (has_ext_ci(path, ".dcm") || has_ext_ci(path, ".dicom")) {
        pr.flavor = Flavor::DICOM;
        pr.reason = "extension .dcm/.dicom";
        dbg_line(dbg, "[IO][probe] -> DICOM by extension");
        return pr;
    }
    if (has_ext_ci(path, ".h5") || has_ext_ci(path, ".hdf5")) {
        pr.flavor = Flavor::HDF5_Unknown;
        pr.reason = "extension .h5/.hdf5";
        dbg_line(dbg, "[IO][probe] -> HDF5 by extension (delegate to DLL/metadata)");
        return pr;
    }

    // No telling by extension: peek magic numbers
    if (is_dicom_magic(path)) {
        pr.flavor = Flavor::DICOM;
        pr.reason = "DICOM magic @128='DICM'";
        dbg_line(dbg, "[IO][probe] -> DICOM by magic");
        return pr;
    }
    if (is_hdf5_magic(path)) {
        pr.flavor = Flavor::HDF5_Unknown;
        pr.reason = "HDF5 magic";
        dbg_line(dbg, "[IO][probe] -> HDF5 by magic (delegate to DLL/metadata)");
        return pr;
    }

    pr.flavor = Flavor::NotHDF5;
    pr.reason = "neither DICOM nor HDF5 signature";
    dbg_line(dbg, "[IO][probe] -> NotHDF5");
    return pr;
}

} // namespace io

// ============================================================
// DCMTK multi-slice reader helpers
// ============================================================
namespace {

static bool dcmtk_read_one_file_u8(const std::string& path,
                                   std::vector<cv::Mat>& frames,
                                   std::string* why)
{
    std::cerr << "[DBG][DICOM][DCMTK] read file: " << path << "\n";

    DcmFileFormat ff;
    if (ff.loadFile(path.c_str()).bad()) {
        if (why) *why = "Cannot load DICOM file";
        std::cerr << "[ERR][DICOM] DcmFileFormat loadFile failed\n";
        return false;
    }

    DicomImage di(path.c_str());
    if (di.getStatus() != EIS_Normal) {
        if (why) *why = "DCMTK: DicomImage status not normal";
        std::cerr << "[ERR][DICOM][DCMTK] DicomImage status=" << di.getStatus() << "\n";
        return false;
    }

    // WindowCenter/Width if present
    double wc = 0.0, ww = 0.0; bool haveWWWC = false;
    {
        OFString sWC, sWW;
        if (ff.getDataset()->findAndGetOFString(DCM_WindowCenter, sWC).good() &&
            ff.getDataset()->findAndGetOFString(DCM_WindowWidth,  sWW).good())
        {
            wc = atof(sWC.c_str());
            ww = atof(sWW.c_str());
            if (ww > 0.0) { haveWWWC = true; di.setWindow(wc, ww); }
            std::cerr << "[DBG][DICOM] WC=" << wc << " WW=" << ww << " haveWWWC=" << haveWWWC << "\n";
        }
    }
    if (!haveWWWC) {
        di.setMinMaxWindow();
        std::cerr << "[DBG][DICOM] Min/Max window\n";
    }

    const int W = (int)di.getWidth();
    const int H = (int)di.getHeight();
    const int F = (int)di.getFrameCount();
    std::cerr << "[DBG][DICOM][DCMTK] dims=" << W << "x" << H << " frames=" << F << "\n";

    const size_t bytesPerFrame = (size_t)W * H;
    std::vector<uint8_t> buf(bytesPerFrame);

    for (int f = 0; f < F; ++f) {
        if (!di.getOutputData(buf.data(), bytesPerFrame, 8 /*bits*/, f)) {
            std::cerr << "[ERR][DICOM] getOutputData(8) failed at frame " << f << "\n";
            continue;
        }
        cv::Mat u8(H, W, CV_8UC1, buf.data());
        frames.push_back(u8.clone());
    }
    return !frames.empty();
}

static bool dcmtk_series_key(const std::string& path,
                             std::string& seriesUID, int& instanceNo, double& ippZ)
{
    DcmFileFormat ff;
    if (ff.loadFile(path.c_str()).bad()) return false;

    OFString s;
    if (ff.getDataset()->findAndGetOFString(DCM_SeriesInstanceUID, s).good())
        seriesUID = s.c_str();
    else
        return false;

    instanceNo = 0;
    ff.getDataset()->findAndGetSint32(DCM_InstanceNumber, instanceNo);

    ippZ = 0.0;
    OFString ipp;
    if (ff.getDataset()->findAndGetOFString(DCM_ImagePositionPatient, ipp).good()) {
        std::string v = ipp.c_str();
        size_t p2 = v.rfind('\\');
        if (p2 != std::string::npos) ippZ = atof(v.c_str() + p2 + 1);
    }
    return true;
}

static bool dcmtk_read_series_u8(const std::string& seedPath,
                                 std::vector<cv::Mat>& frames,
                                 std::string* why)
{
    fs::path p(seedPath);
    const fs::path dir = p.parent_path();
    if (dir.empty() || !fs::exists(dir)) {
        if (why) *why = "DCMTK series: directory not found";
        return false;
    }

    std::string targetUID; int dummyN = 0; double dummyZ = 0.0;
    if (!dcmtk_series_key(seedPath, targetUID, dummyN, dummyZ)) {
        if (why) *why = "DCMTK: cannot read SeriesInstanceUID from seed";
        return false;
    }
    std::cerr << "[DBG][DICOM][SERIES] seed UID=" << targetUID << " dir=" << dir.string() << "\n";

    struct Item { std::string path; int inst; double z; };
    std::vector<Item> items;

    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        const std::string f = e.path().string();

        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
        if (!(ext == ".dcm" || ext.empty())) continue;

        std::string uid; int inst=0; double z=0.0;
        if (!dcmtk_series_key(f, uid, inst, z)) continue;
        if (uid == targetUID) items.push_back({f, inst, z});
    }

    if (items.empty()) {
        if (why) *why = "DCMTK series: no siblings found with same SeriesInstanceUID";
        return false;
    }

    std::stable_sort(items.begin(), items.end(),
                     [](const Item& a, const Item& b) {
                         if (a.inst != 0 || b.inst != 0) return a.inst < b.inst;
                         return a.z < b.z;
                     });

    std::cerr << "[DBG][DICOM][SERIES] files in series=" << items.size() << "\n";

    std::vector<cv::Mat> out; out.reserve(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        std::vector<cv::Mat> tmp;
        if (!dcmtk_read_one_file_u8(items[i].path, tmp, why) || tmp.empty()) {
            std::cerr << "[WRN][DICOM] skip unreadable: " << items[i].path << "\n";
            continue;
        }
        out.insert(out.end(), tmp.begin(), tmp.end());
    }
    frames.swap(out);
    return !frames.empty();
}

} // anon

// ============================================================
// io namespace wrappers (public API)
// ============================================================
namespace io {

bool read_dicom_frames_gray8(const std::string& path,
                             std::vector<cv::Mat>& out,
                             std::string* why)
{
    std::cerr << "[DBG][DICOM] read_dicom_frames_gray8: " << path << "\n";
    out.clear();

    // 1) Multi-frame in the file
    {
        std::vector<cv::Mat> mf;
        if (dcmtk_read_one_file_u8(path, mf, why) && !mf.empty()) {
            std::cerr << "[DBG][DICOM] file frames=" << mf.size() << "\n";
            out = std::move(mf);
            return true;
        }
    }
    // 2) Expand to series
    {
        std::vector<cv::Mat> series;
        std::string why2;
        if (dcmtk_read_series_u8(path, series, &why2) && !series.empty()) {
            std::cerr << "[DBG][DICOM] series slices=" << series.size() << "\n";
            out = std::move(series);
            return true;
        }
        if (why && !why2.empty()) *why = why2;
    }

    if (why && why->empty()) *why = "No frames found in file or series.";
    return false;
}

bool read_dicom_gray8(const std::string& path, cv::Mat& out8, std::string* why)
{
    std::vector<cv::Mat> frames;
    if (!read_dicom_frames_gray8(path, frames, why) || frames.empty()) {
        if (why && why->empty()) *why = "DICOM read failed (no frames).";
        return false;
    }
    out8 = frames.front().clone();
    std::cerr << "[DBG][DICOM] read_dicom_gray8 -> dims="
              << out8.cols << "x" << out8.rows << "\n";
    return true;
}

bool read_dicom_frames_u16(const std::string& path,
                           std::vector<cv::Mat>& out16,
                           std::string* why)
{
    out16.clear();

    DcmFileFormat ff;
    if (ff.loadFile(path.c_str()).bad()) { if (why) *why = "Cannot load file"; return false; }

    DicomImage di(path.c_str());
    if (di.getStatus() != EIS_Normal) { if (why) *why = "Bad DicomImage"; return false; }

    double slope = 1.0, intercept = 0.0;
    ff.getDataset()->findAndGetFloat64(DCM_RescaleSlope,     slope);
    ff.getDataset()->findAndGetFloat64(DCM_RescaleIntercept, intercept);
    std::cerr << "[DBG][DICOM] RS=" << slope << " RI=" << intercept << "\n";

    const int W = (int)di.getWidth();
    const int H = (int)di.getHeight();
    const int F = (int)di.getFrameCount();

    const size_t bytesPerFrame = (size_t)W * H * 2;
    std::vector<uint16_t> buf((size_t)W * H);

    for (int f = 0; f < F; ++f) {
        if (!di.getOutputData(buf.data(), bytesPerFrame, 16 /*bits*/, f)) {
            std::cerr << "[ERR][DICOM] getOutputData(16) failed at frame " << f << "\n";
            continue;
        }
        cv::Mat raw16(H, W, CV_16UC1);
        for (int y = 0; y < H; ++y) {
            const uint16_t* src = buf.data() + (size_t)y * W;
            uint16_t* dst = raw16.ptr<uint16_t>(y);
            for (int x = 0; x < W; ++x) {
                double v = (double)src[x] * slope + intercept;
                if (v < 0.0) v = 0.0;
                if (v > 65535.0) v = 65535.0;
                dst[x] = static_cast<uint16_t>(v + 0.5);
            }
        }
        out16.push_back(raw16.clone());
    }
    std::cerr << "[DBG][DICOM] read_dicom_frames_u16 -> frames=" << out16.size() << "\n";
    return !out16.empty();
}

bool write_png(const std::string& path_utf8, const cv::Mat& img8, std::string* why)
{
    try {
        if (img8.empty()) { if (why) *why = "empty image"; return false; }
        std::vector<int> params;
        if (!cv::imwrite(path_utf8, img8, params)) {
            if (why) *why = "cv::imwrite returned false";
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        if (why) *why = std::string("cv::imwrite exception: ") + e.what();
        return false;
    }
}

bool write_dicom_sc_gray8(const std::string& path, const cv::Mat& img8, std::string* why)
{
    if (why) *why = "DICOM write not built (requires DCMTK/GDCM).";
    return false;
}

bool read_dicom_basic_meta(const std::string& path, DicomMeta& out, std::string* why)
{
    std::cerr << "[DBG][DICOM][META] read: " << path << "\n";

    DcmFileFormat ff;
    OFCondition st = ff.loadFile(path.c_str());
    if (st.bad()) {
        if (why) *why = "Cannot load file";
        std::cerr << "[ERR][DICOM][META] loadFile failed: " << st.text() << "\n";
        return false;
    }
    DcmDataset* ds = ff.getDataset();

    auto getS = [&](const DcmTagKey& key) -> std::string {
        OFString s; if (ds->findAndGetOFString(key, s).good()) return s.c_str();
        return {};
    };
    auto getF = [&](const DcmTagKey& key) -> std::string {
        double v=0.0; if (ds->findAndGetFloat64(key, v).good()) {
            char buf[64]; snprintf(buf, sizeof(buf), "%.3f", v);
            return buf;
        }
        return {};
    };

    out.manufacturer      = getS(DCM_Manufacturer);                // 0008,0070
    out.modelName         = getS(DCM_ManufacturerModelName);       // 0008,1090
    out.softwareVersions  = getS(DCM_SoftwareVersions);            // 0018,1020
    out.institutionName   = getS(DCM_InstitutionName);             // 0008,0080
    out.seriesDescription = getS(DCM_SeriesDescription);           // 0008,103E
    out.patientName       = getS(DCM_PatientName);                 // 0010,0010
    out.patientID         = getS(DCM_PatientID);                   // 0010,0020
    out.studyDate         = getS(DCM_StudyDate);                   // 0008,0020
    out.studyTime         = getS(DCM_StudyTime);                   // 0008,0030

    // MagneticFieldStrength may be DS/FD; try numeric first, then string
    out.B0T = getF(DCM_MagneticFieldStrength);                     // 0018,0087
    if (out.B0T.empty()) out.B0T = getS(DCM_MagneticFieldStrength);

    // New: TR/TE/TI (ms)
    out.tr_ms = getF(DCM_RepetitionTime); if (out.tr_ms.empty()) out.tr_ms = getS(DCM_RepetitionTime);   // 0018,0080
    out.te_ms = getF(DCM_EchoTime);       if (out.te_ms.empty()) out.te_ms = getS(DCM_EchoTime);         // 0018,0081
    out.ti_ms = getF(DCM_InversionTime);  if (out.ti_ms.empty()) out.ti_ms = getS(DCM_InversionTime);    // 0018,0082

    std::cerr << "[DBG][DICOM][META] Manufacturer='" << out.manufacturer
              << "' Model='" << out.modelName
              << "' B0T='" << out.B0T
              << "' TR/TE/TI(ms)='" << out.tr_ms << "/" << out.te_ms << "/" << out.ti_ms
              << "'\n";
    return true;
}


} // namespace io

// ============================================================
// HDF5 helpers (for ISMRMRD XML in fastMRI/ISMRMRD files)
// ============================================================
namespace {

// tiny dbg aliases for this block
static void h5_dbg(const char* s)  { std::cerr << s << "\n"; }
static void h5_dbg2(const std::string& a, const std::string& b) { std::cerr << a << b << "\n"; }

// Read FIRST string from a dataset of strings (handles VAR-LEN and FIXED-LEN)
static bool h5_read_first_string(H5::DataSet& ds, std::string& out)
{
    try {
        H5::DataSpace space = ds.getSpace();
        int nd = space.getSimpleExtentNdims();
        hsize_t dims[H5S_MAX_RANK] = {0};
        hsize_t n = 1;
        if (nd > 0) {
            space.getSimpleExtentDims(dims, nullptr);
            for (int i = 0; i < nd; ++i) n *= dims[i];
        }
        if (n == 0) { std::cerr << "[H5][meta] dataset is empty\n"; return false; }

        H5::DataType t = ds.getDataType();
        if (t.getClass() != H5T_STRING) {
            std::cerr << "[H5][meta] dataset is not string type\n";
            return false;
        }

        const bool isVarLen = H5Tis_variable_str(t.getId()) > 0;

        if (isVarLen) {
            // --- VAR-LEN path ---
            H5::StrType vl(H5::PredType::C_S1, H5T_VARIABLE);
            std::vector<char*> arr(static_cast<size_t>(n), nullptr);

            ds.read(arr.data(), vl); // HDF5 allocates each char* internally
            for (size_t i = 0; i < arr.size(); ++i) {
                if (arr[i]) { out.assign(arr[i]); break; }
            }

            // Reclaim all var-len memory that HDF5 allocated. Do NOT free() afterwards.
            hid_t space_id = space.getId();
            hid_t type_id  = vl.getId();
            H5Dvlen_reclaim(type_id, space_id, H5P_DEFAULT, arr.data());

            std::cerr << "[H5][meta] VARLEN read; first string len="
                      << out.size() << "\n";
            return !out.empty();
        } else {
            // --- FIXED-LEN path ---
            H5::StrType st = ds.getStrType();
            const size_t len = st.getSize();
            if (len == 0) { std::cerr << "[H5][meta] fixed string length is 0\n"; return false; }

            std::vector<char> buf(static_cast<size_t>(n) * len);
            ds.read(buf.data(), st);

            // take first fixed-length string and trim trailing nulls/whitespace
            std::string s(buf.data(), buf.data() + len);
            while (!s.empty() &&
                   (s.back() == '\0' || s.back() == ' ' || s.back() == '\n' ||
                    s.back() == '\r'  || s.back() == '\t')) {
                s.pop_back();
            }

            out = std::move(s);
            std::cerr << "[H5][meta] FIXED read; first string len="
                      << out.size() << "\n";
            return !out.empty();
        }
    } catch (const H5::Exception& e) {
        std::cerr << "[H5][meta] h5_read_first_string exception: "
                  << e.getDetailMsg() << "\n";
        return false;
    }
}


// Try open a dataset by name and read first string
static bool h5_try_dataset_first_string(H5::H5File& f, const char* name, std::string& out)
{
    try {
        H5::DataSet ds = f.openDataSet(name);
        h5_dbg2("[H5][meta] found dataset: ", name);
        return h5_read_first_string(ds, out);
    } catch (...) {
        return false;
    }
}

// Try read a root attribute as a var-len string
static bool h5_try_root_attr_string(H5::H5File& f, const char* attrName, std::string& out)
{
    try {
        H5::Group root = f.openGroup("/");
        if (!root.attrExists(attrName)) return false;
        H5::Attribute a = root.openAttribute(attrName);
        H5::StrType vl(H5::PredType::C_S1, H5T_VARIABLE);
        char* s = nullptr;
        a.read(vl, &s);
        if (s) { out.assign(s); free(s); }
        h5_dbg2("[H5][meta] found root attribute: ", attrName);
        return !out.empty();
    } catch (...) {
        return false;
    }
}

// Extract a child text safely (empty if missing)
static std::string xtext(pugi::xml_node n, const char* child)
{
    if (!n) return {};
    auto c = n.child(child);
    if (!c) return {};
    const char* v = c.child_value();
    return v ? std::string(v) : std::string();
}

} // anon

// ------------------------------------------------------------
// HDF5/ISMRMRD metadata (fastMRI & generic ISMRMRD)
// ------------------------------------------------------------
namespace io {
bool io::read_hdf5_ismrmrd_meta(const std::string& path, DicomMeta& out, std::string* why)
{
    try {
        h5_dbg2("[H5][meta] open file: ", path);
        H5::H5File f(path, H5F_ACC_RDONLY);

        std::string xml;

        // Try common locations (fastMRI & ISMRMRD)
        if (!h5_try_dataset_first_string(f, "ismrmrd_header", xml)) {
            if (!h5_try_dataset_first_string(f, "/dataset/xml", xml) &&
                !h5_try_dataset_first_string(f, "dataset/xml", xml)) {
                // Last resort: root attribute
                h5_try_root_attr_string(f, "ismrmrd_header", xml);
            }
        }

        if (xml.empty()) {
            if (why) *why = "ISMRMRD XML not found in HDF5 (tried datasets: 'ismrmrd_header', '/dataset/xml', and root attr 'ismrmrd_header').";
            h5_dbg("[H5][meta] XML not found");
            return false;
        }

        h5_dbg("[H5][meta] XML found; parsing with pugixml...");
        pugi::xml_document doc;
        pugi::xml_parse_result pr = doc.load_string(xml.c_str());
        if (!pr) {
            if (why) *why = std::string("pugixml parse error: ") + pr.description();
            h5_dbg2("[H5][meta] XML parse error: ", pr.description());
            return false;
        }

        // ismrmrdHeader/acquisitionSystemInformation/... and /sequenceParameters/...
        auto root = doc.child("ismrmrdHeader");
        auto asi  = root.child("acquisitionSystemInformation");
        auto seq  = root.child("sequenceParameters");

        out.manufacturer     = xtext(asi, "systemVendor");
        out.modelName        = xtext(asi, "systemModel");
        out.B0T              = xtext(asi, "systemFieldStrength_T");
        out.institutionName  = xtext(asi, "institutionName");
        out.softwareVersions = xtext(asi, "systemVendorIdentifier"); // best-effort

        // Save timing (ms) into the struct so the controller can display it
        out.tr_ms = xtext(seq, "TR");
        out.te_ms = xtext(seq, "TE");
        out.ti_ms = xtext(seq, "TI");

        if (!out.tr_ms.empty() || !out.te_ms.empty() || !out.ti_ms.empty()) {
            h5_dbg2("[H5][meta] TR/TE/TI(ms): ", (out.tr_ms + "/" + out.te_ms + "/" + out.ti_ms));
        }

        h5_dbg2("[H5][meta] Manufacturer: ", out.manufacturer);
        h5_dbg2("[H5][meta] Model: ", out.modelName);
        h5_dbg2("[H5][meta] Field (T): ", out.B0T);
        h5_dbg2("[H5][meta] Institution: ", out.institutionName);
        return true;
    } catch (const H5::Exception& e) {
        if (why) *why = std::string("HDF5 exception: ") + e.getDetailMsg();
        h5_dbg2("[H5][meta] HDF5 exception: ", e.getDetailMsg());
        return false;
    } catch (const std::exception& e) {
        if (why) *why = std::string("std::exception: ") + e.what();
        h5_dbg2("[H5][meta] std::exception: ", e.what());
        return false;
    }
}


} // namespace io

