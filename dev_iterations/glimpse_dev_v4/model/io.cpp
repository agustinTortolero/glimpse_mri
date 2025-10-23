// model/io.cpp
#include "io.hpp"

#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

// ---------- CRT check toggle (Debug-only, clarity over perf) ----------
#ifndef DICOM_CRT_HEAVY_CHECKS
#define DICOM_CRT_HEAVY_CHECKS 0
#endif

#if defined(_MSC_VER) && defined(_DEBUG) && DICOM_CRT_HEAVY_CHECKS
#define CRT_CHECK() do { \
if (!_CrtCheckMemory()) { \
        std::cerr << "[CRT][ERR] _CrtCheckMemory FAILED at line " << __LINE__ << "\n"; \
} \
} while(0)
#else
#define CRT_CHECK() do{}while(0)
#endif

// ---------- Diagnostic toggles ----------
// 0 = normal; 1 = early return (for experiments)
#ifndef DCMTK_SKIP_DI_DESTRUCTOR_ON_SUCCESS
#define DCMTK_SKIP_DI_DESTRUCTOR_ON_SUCCESS 0
#endif
// 0 = normal; 1 = ALSO skip ~DcmFileFormat on success (diagnostic)
#ifndef DCMTK_SKIP_FF_DESTRUCTOR_ON_SUCCESS
#define DCMTK_SKIP_FF_DESTRUCTOR_ON_SUCCESS 0
#endif

// If 1 → skip DicomImage path and use raw PixelData fallback only.
#ifndef DCMTK_FORCE_RAW_PIXEL_FALLBACK
#define DCMTK_FORCE_RAW_PIXEL_FALLBACK 0
#endif

// Write a diagnostic PNG of the first decoded frame to the temp directory
#ifndef IO_DEBUG_DUMP_FIRST_PNG
#define IO_DEBUG_DUMP_FIRST_PNG 1
#endif

// ---------------- OpenCV ----------------
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// ---------------- STL ----------------
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <cmath>   // for sqrt in oblique ordering

namespace fs = std::filesystem;

// ---------------- DCMTK ----------------
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmdata/dcuid.h>
#include <dcmtk/dcmdata/dcistrmf.h>
#include <dcmtk/dcmdata/dcostrmf.h>
#include <dcmtk/dcmdata/dcrledrg.h> // RLE decoder
#include <dcmtk/dcmimgle/dcmimage.h>
#include <dcmtk/dcmjpeg/djdecode.h> // JPEG
#include <dcmtk/dcmjpls/djdecode.h> // JPEG-LS
#ifdef WITH_DCMTK_DCMJ2K
#include <dcmtk/dcmj2k/djdecode.h>  // JPEG 2000 (if built)
#endif

// ======== DICOM Secondary Capture (grayscale, 8-bit, single frame) ========
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcuid.h>        // UID_*, dcmGenerateUniqueIdentifier
#include <dcmtk/dcmdata/dcostrmb.h>
#include <dcmtk/dcmdata/dcmetinf.h>

// ---------------- HDF5 + pugixml (for ISMRMRD XML) ----------------
#include <H5Cpp.h>
#include <hdf5.h>   // H5Dvlen_reclaim
#include <pugixml.hpp>

// ---------------------------------
// Small helpers (debug logging)
// ---------------------------------
static inline void dbg_line(std::string* dbg, const std::string& s) {
    if (dbg) dbg->append(s + "\n");
    std::cerr << s << "\n";
}
static inline std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}
static inline bool has_ext_ci(const std::string& path, const char* ext) {
    const auto e = to_lower_copy(fs::path(path).extension().string());
    return e == to_lower_copy(ext);
}
static bool is_dicom_magic(const std::string& path) {
    // DICOM: "DICM" at offset 128
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(128, std::ios::beg);
    char tag[4] = {0,0,0,0};
    f.read(tag, 4);
    return (f.gcount() == 4 && tag[0]=='D' && tag[1]=='I' && tag[2]=='C' && tag[3]=='M');
}
static bool is_hdf5_magic(const std::string& path) {
    // HDF5 magic: 0x89 H D F \r \n 0x1A \n
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    unsigned char sig[8] = {};
    f.read(reinterpret_cast<char*>(sig), 8);
    if (f.gcount() != 8) return false;
    const unsigned char ref[8] = { 0x89, 'H', 'D', 'F', 0x0d, 0x0a, 0x1a, 0x0a };
    for (int i = 0; i < 8; ++i) if (sig[i] != ref[i]) return false;
    return true;
}

// ---------------------------------
// DCMTK global codec init
// ---------------------------------
namespace { bool g_dcmtk_inited = false; }

namespace io {

void dcmtk_global_init() {
    if (g_dcmtk_inited) return;
    std::cerr << "[DCMTK][DBG] Codecs registering (JPEG/JLS/RLE"
#ifdef WITH_DCMTK_DCMJ2K
                 "/J2K"
#endif
                 ")\n";
    DJDecoderRegistration::registerCodecs();
    DJLSDecoderRegistration::registerCodecs();
    DcmRLEDecoderRegistration::registerCodecs();
#ifdef WITH_DCMTK_DCMJ2K
    DcmJ2kDecoderRegistration::registerCodecs();
#endif
    g_dcmtk_inited = true;
    std::cerr << "[DCMTK][DBG] Codecs registered\n";
}

void dcmtk_global_shutdown() {
    if (!g_dcmtk_inited) return;
    std::cerr << "[DCMTK][DBG] Codecs unregistering\n";
#ifdef WITH_DCMTK_DCMJ2K
    DcmJ2kDecoderRegistration::cleanup();
#endif
    DcmRLEDecoderRegistration::cleanup();
    DJLSDecoderRegistration::cleanup();
    DJDecoderRegistration::cleanup();
    g_dcmtk_inited = false;
}

// ---------------------------------
// Probe by extension/magic (UI hint)
// ---------------------------------
ProbeResult probe(const std::string& path, std::string* dbg) {
    dbg_line(dbg, "[IO][probe] path=" + path);

    if (has_ext_ci(path, ".dcm") || has_ext_ci(path, ".dicom") || is_dicom_magic(path)) {
        dbg_line(dbg, "[IO][probe] -> DICOM by extension/magic");
        return {Flavor::DICOM, ""};
    }
    if (has_ext_ci(path, ".h5") || has_ext_ci(path, ".hdf5") || is_hdf5_magic(path)) {
        dbg_line(dbg, "[IO][probe] -> HDF5 by extension (delegate to DLL/metadata)");
        return {Flavor::HDF5_Unknown, ""};
    }

    // Fallback: treat as HDF5_Unknown to avoid using a non-existent Flavor::Unknown
    dbg_line(dbg, "[IO][probe] -> Unknown (treat as HDF5_Unknown for UI)");
    return {Flavor::HDF5_Unknown, ""};
}

} // namespace io

// ---------------------------------
// Internal DICOM helpers (anon namespace)
// ---------------------------------
namespace {

// ------------------------------------------------------------
// Oblique-safe ordering key:
// - Returns SeriesInstanceUID (uid), InstanceNumber (inst),
//   and projection distance zproj = dot(IPP, normal),
//   where normal = normalize(row × col) from IOP.
// - Falls back to IPP.z or 0.0 if tags missing.
// ------------------------------------------------------------
static bool dcmtk_series_key(const std::string& path,
                             std::string& uid, int& inst, double& zproj)
{
    std::cerr << "[DBG][DICOM][SERIES_KEY] ENTER path=" << path << "\n";
    DcmFileFormat ff;
    if (ff.loadFile(path.c_str()).bad()) {
        std::cerr << "[DBG][DICOM][SERIES_KEY] loadFile failed\n";
        return false;
    }
    DcmDataset* ds = ff.getDataset();
    if (!ds) { std::cerr << "[DBG][DICOM][SERIES_KEY] no dataset\n"; return false; }

    OFString suid;
    if (ds->findAndGetOFString(DCM_SeriesInstanceUID, suid).bad() || suid.empty()) {
        std::cerr << "[DBG][DICOM][SERIES_KEY] missing SeriesInstanceUID\n";
        return false;
    }
    uid = suid.c_str();

    Sint32 instNum = 0;
    if (ds->findAndGetSint32(DCM_InstanceNumber, instNum).good())
        inst = static_cast<int>(instNum);
    else
        inst = 0;

    // Defaults
    zproj = 0.0;

    // Parse IOP (6 values)
    double r[3] = {0,0,0}, c[3] = {0,0,0};
    bool haveIOP = false;
    {
        OFString iop;
        if (ds->findAndGetOFStringArray(DCM_ImageOrientationPatient, iop).good()) {
            std::string s = iop.c_str();
            double v[6] = {0,0,0,0,0,0};
            int k = 0; size_t start = 0;
            while (k < 6) {
                size_t p = s.find('\\', start);
                std::string tok = s.substr(start, (p == std::string::npos) ? std::string::npos : p - start);
                try { v[k++] = std::stod(tok); } catch (...) { v[k-1] = 0.0; }
                if (p == std::string::npos) break;
                start = p + 1;
            }
            if (k >= 6) {
                r[0]=v[0]; r[1]=v[1]; r[2]=v[2];
                c[0]=v[3]; c[1]=v[4]; c[2]=v[5];
                haveIOP = true;
            }
        }
    }

    // Parse IPP (3 values)
    double pnt[3] = {0,0,0};
    bool haveIPP = false;
    {
        OFString ipp;
        if (ds->findAndGetOFStringArray(DCM_ImagePositionPatient, ipp).good()) {
            std::string s = ipp.c_str();
            double v[3] = {0,0,0};
            int k = 0; size_t start = 0;
            while (k < 3) {
                size_t p = s.find('\\', start);
                std::string tok = s.substr(start, (p == std::string::npos) ? std::string::npos : p - start);
                try { v[k++] = std::stod(tok); } catch (...) { v[k-1] = 0.0; }
                if (p == std::string::npos) break;
                start = p + 1;
            }
            if (k >= 3) {
                pnt[0]=v[0]; pnt[1]=v[1]; pnt[2]=v[2];
                haveIPP = true;
            }
        }
    }

    if (haveIOP && haveIPP) {
        // normal = normalize(row × col)
        double n[3] = {
            r[1]*c[2] - r[2]*c[1],
            r[2]*c[0] - r[0]*c[2],
            r[0]*c[1] - r[1]*c[0]
        };
        double len = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
        if (len > 0.0) { n[0]/=len; n[1]/=len; n[2]/=len; }
        zproj = pnt[0]*n[0] + pnt[1]*n[1] + pnt[2]*n[2];
        std::cerr << "[DBG][DICOM][SERIES_KEY] proj=dot(IPP,normal)=" << zproj
                  << " inst=" << inst << "\n";
    } else if (haveIPP) {
        // fallback: use IPP.z as secondary key
        zproj = pnt[2];
        std::cerr << "[DBG][DICOM][SERIES_KEY] fallback proj=IPP.z=" << zproj
                  << " inst=" << inst << "\n";
    } else {
        std::cerr << "[DBG][DICOM][SERIES_KEY] no IOP/IPP; proj=0 inst=" << inst << "\n";
    }

    std::cerr << "[DBG][DICOM][SERIES_KEY] EXIT uid=" << uid
              << " inst=" << inst << " proj=" << zproj << "\n";
    return true;
}

// ------------------------------
// Single-file → frames (8-bit), robust:
// 1) Try DicomImage (handles compressed transfer syntaxes)
// 2) Fallback to raw PixelData path for uncompressed mono
// ------------------------------
static bool dcmtk_read_one_file_u8(const std::string& path,
                                   std::vector<cv::Mat>& frames,
                                   std::string* why)
{
    std::cerr << "[DBG][DICOM][DCMTK] ENTER read_u8: " << path << "\n";
    io::dcmtk_global_init();
    frames.clear();

#if !DCMTK_FORCE_RAW_PIXEL_FALLBACK
    // ---------- 1) Try DicomImage first (robust, handles compression) ----------
    {
        std::cerr << "[DBG][DICOM][DCMTK] trying DicomImage path...\n";
        std::unique_ptr<DicomImage> di(new DicomImage(path.c_str()));
        const bool di_ok = (di && di->getStatus() == EIS_Normal);
        std::cerr << "[DBG][DICOM][DCMTK] DicomImage status="
                  << (di ? (int)di->getStatus() : -1)
                  << " ok=" << (di_ok ? "1" : "0") << "\n";

        if (di_ok) {
            const unsigned long w = di->getWidth();
            const unsigned long h = di->getHeight();
            const unsigned long nFrames = std::max<unsigned long>(1, di->getFrameCount());
            const OFBool isMono = di->isMonochrome();

            std::cerr << "[DBG][DICOM][DCMTK] DicomImage dims=" << w << "x" << h
                      << " frames=" << nFrames
                      << " isMono=" << (isMono ? "1" : "0") << "\n";

            // Ensure a contrast window so output spans 0..255
            di->setMinMaxWindow();

            frames.reserve(nFrames);
            for (unsigned long f = 0; f < nFrames; ++f) {
                const void* out = di->getOutputData(8, f, 0 /*planar*/);
                if (!out) {
                    std::cerr << "[WRN][DICOM][DCMTK] getOutputData returned null at frame " << f << "\n";
                    continue;
                }
                cv::Mat m((int)h, (int)w, CV_8UC1);
                std::memcpy(m.data, out, (size_t)w * (size_t)h);
                frames.emplace_back(std::move(m));
                std::cerr << "[DBG][DICOM][DCMTK] DicomImage frame " << (f+1) << "/" << nFrames << " ok\n";
            }

#if IO_DEBUG_DUMP_FIRST_PNG
            if (!frames.empty()) {
                try {
                    auto outPng = (std::filesystem::temp_directory_path() / "glimpse_diag_first.png").string();
                    cv::imwrite(outPng, frames[0]);
                    std::cerr << "[DBG][DICOM][MIN] wrote diagnostic PNG: " << outPng << "\n";
                } catch (...) {
                    std::cerr << "[WRN][DICOM][MIN] PNG dump failed\n";
                }
            }
#endif
            std::cerr << "[DBG][DICOM][DCMTK] EXIT read_u8 via DicomImage, frames=" << frames.size() << "\n";
            return !frames.empty();
        }

        std::cerr << "[WRN][DICOM][DCMTK] DicomImage decode failed; will try raw PixelData if uncompressed\n";
    }
#else
    std::cerr << "[DBG][DICOM][DCMTK] DicomImage path DISABLED by DCMTK_FORCE_RAW_PIXEL_FALLBACK\n";
#endif

    // ---------- 2) Fallback: raw PixelData path (uncompressed mono only) ----------
    std::cerr << "[DBG][DICOM][DCMTK] raw PixelData fallback...\n";
    auto ff = std::make_unique<DcmFileFormat>();
    OFCondition st = ff->loadFile(path.c_str());
    if (st.bad()) {
        if (why) *why = std::string("Cannot load DICOM file: ") + st.text();
        std::cerr << "[ERR][DICOM] loadFile failed: " << st.text() << "\n";
        return false;
    }
    DcmDataset* ds = ff->getDataset();
    if (!ds) {
        if (why) *why = "No dataset";
        std::cerr << "[ERR][DICOM] No dataset\n";
        return false;
    }

    // If transfer syntax is encapsulated (compressed), bail from raw path
    const E_TransferSyntax xfer = ds->getOriginalXfer();
    const DcmXfer xf(xfer);
    const OFBool isEnc = xf.isEncapsulated();
    std::cerr << "[DBG][DICOM][RAW] TransferSyntax=" << xf.getXferName()
              << " encapsulated=" << (isEnc ? "1" : "0") << "\n";
    if (isEnc) {
        if (why) *why = "Compressed transfer syntax; DicomImage decode failed";
        std::cerr << "[ERR][DICOM][RAW] Encapsulated TS -> aborting raw decode\n";
        return false;
    }

    Uint16 rows=0, cols=0, bitsAlloc=0, spp=0;
    ds->findAndGetUint16(DCM_Rows, rows);
    ds->findAndGetUint16(DCM_Columns, cols);
    ds->findAndGetUint16(DCM_BitsAllocated, bitsAlloc);
    ds->findAndGetUint16(DCM_SamplesPerPixel, spp);
    if (!rows || !cols) {
        if (why) *why = "Missing Rows/Columns";
        std::cerr << "[ERR][DICOM] Missing Rows/Columns\n";
        return false;
    }
    if (!bitsAlloc) bitsAlloc = 16;
    if (!spp)       spp       = 1;

    // NumberOfFrames (optional; defaults to 1)
    uint32_t nFrames = 1;
    {
        OFString nf;
        if (ds->findAndGetOFString(DCM_NumberOfFrames, nf).good()) {
            try {
                int tmp = std::max(1, std::stoi(nf.c_str()));
                nFrames = static_cast<uint32_t>(tmp);
            } catch(...) { /* keep 1 */ }
        }
    }

    // PixelRepresentation (0 = unsigned, 1 = signed), used for 16-bit scaling
    Uint16 pixRep = 0;
    ds->findAndGetUint16(DCM_PixelRepresentation, pixRep);

    std::cerr << "[DBG][DICOM][MIN] rows=" << rows
              << " cols=" << cols
              << " BitsAllocated=" << bitsAlloc
              << " SamplesPerPixel=" << spp
              << " NumberOfFrames=" << nFrames
              << " PixelRepresentation=" << pixRep
              << "\n";

    // PixelData
    DcmElement* elem = nullptr;
    st = ds->findAndGetElement(DCM_PixelData, elem, /*searchIntoSub=*/true);
    if (st.bad() || !elem) {
        if (why) *why = "No PixelData";
        std::cerr << "[ERR][DICOM] Cannot find PixelData: " << st.text() << "\n";
        return false;
    }

    const uint32_t framePixels = static_cast<uint32_t>(rows) * static_cast<uint32_t>(cols);
    frames.clear();

    // ---------------- 8-bit mono ----------------
    if (spp == 1 && bitsAlloc == 8) {
        Uint8* base = nullptr;
        st = elem->getUint8Array(base);
        if (st.bad() || !base) {
            if (why) *why = "getUint8Array failed";
            std::cerr << "[ERR][DICOM] getUint8Array failed: " << st.text() << "\n";
            return false;
        }
        const Uint32 totalBytes = elem->getLength();
        const Uint64 expected   = static_cast<Uint64>(framePixels) * static_cast<Uint64>(nFrames);
        if (totalBytes < expected) {
            std::cerr << "[WRN][DICOM][MIN] 8-bit PixelData smaller than expected ("
                      << totalBytes << " < " << expected << "); forcing nFrames=1\n";
            nFrames = 1;
        }

        frames.reserve(nFrames);
        for (uint32_t f = 0; f < nFrames; ++f) {
            const Uint8* src = base + static_cast<size_t>(f) * framePixels;
            cv::Mat m(rows, cols, CV_8UC1);
            std::memcpy(m.data, src, framePixels);
            frames.emplace_back(std::move(m));
            std::cerr << "[DBG][DICOM][MIN] produced frame " << (f+1) << "/" << nFrames
                      << " (8-bit mono)\n";
        }
    }
    // ---------------- 16-bit mono → scale to 8-bit ----------------
    else if (spp == 1 && bitsAlloc == 16) {
        Uint16* base16 = nullptr;
        st = elem->getUint16Array(base16);
        if (st.bad() || !base16) {
            if (why) *why = "getUint16Array failed";
            std::cerr << "[ERR][DICOM] getUint16Array failed: " << st.text() << "\n";
            return false;
        }
        const Uint32 totalWords = elem->getLength() / 2;
        const Uint64 expected   = static_cast<Uint64>(framePixels) * static_cast<Uint64>(nFrames);
        if (totalWords < expected) {
            std::cerr << "[WRN][DICOM][MIN] 16-bit PixelData smaller than expected ("
                      << totalWords << " < " << expected << "); forcing nFrames=1\n";
            nFrames = 1;
        }

        frames.reserve(nFrames);
        for (uint32_t f = 0; f < nFrames; ++f) {
            const Uint16* src = base16 + static_cast<size_t>(f) * framePixels;

            // Per-frame min/max for better visual contrast
            Uint16 vmin = std::numeric_limits<Uint16>::max();
            Uint16 vmax = 0;
            for (uint32_t i = 0; i < framePixels; ++i) {
                const Uint16 v = src[i];
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
            cv::Mat m(rows, cols, CV_8UC1);
            uint8_t* dst = m.data;
            if (vmax <= vmin) {
                std::memset(dst, 0, framePixels);
            } else {
                const int den = (int)vmax - (int)vmin;
                for (uint32_t i = 0; i < framePixels; ++i) {
                    const int out = ((int)src[i] - (int)vmin) * 255 / den;
                    dst[i] = static_cast<uint8_t>(std::clamp(out, 0, 255));
                }
            }

            frames.emplace_back(std::move(m));
            std::cerr << "[DBG][DICOM][MIN] produced frame " << (f+1) << "/" << nFrames
                      << " (scaled from 16-bit)\n";
        }
    }
    else {
        if (why) *why = "Unsupported PixelData layout (only mono 8/16-bit)";
        std::cerr << "[ERR][DICOM][MIN] unsupported: BitsAllocated=" << bitsAlloc
                  << " SamplesPerPixel=" << spp << "\n";
        return false;
    }

#if IO_DEBUG_DUMP_FIRST_PNG
    if (!frames.empty()) {
        try {
            auto outPng = (std::filesystem::temp_directory_path() / "glimpse_diag_first.png").string();
            cv::imwrite(outPng, frames[0]);
            std::cerr << "[DBG][DICOM][MIN] wrote diagnostic PNG: " << outPng << "\n";
        } catch (...) {
            std::cerr << "[WRN][DICOM][MIN] PNG dump failed\n";
        }
    }
#endif

    std::cerr << "[DBG][DICOM][DCMTK] EXIT read_u8 via raw path, frames=" << frames.size() << "\n";
    return !frames.empty();
}

// ------------------------------
// Single-file → frames (16-bit)
// ------------------------------
static bool dcmtk_read_one_file_u16(const std::string& path,
                                    std::vector<cv::Mat>& frames16,
                                    std::string* why)
{
    std::cerr << "[DBG][DICOM][DCMTK] read file (multi-frame aware, 16-bit): " << path << "\n";
    io::dcmtk_global_init();

    auto ff = std::make_unique<DcmFileFormat>();
    OFCondition st = ff->loadFile(path.c_str());
    if (st.bad()) {
        if (why) *why = std::string("Cannot load DICOM file: ") + st.text();
        std::cerr << "[ERR][DICOM] loadFile failed: " << st.text() << "\n";
        return false;
    }
    DcmDataset* ds = ff->getDataset();
    if (!ds) {
        if (why) *why = "No dataset";
        std::cerr << "[ERR][DICOM] No dataset\n";
        return false;
    }

    Uint16 rows=0, cols=0, bitsAlloc=0, spp=0;
    ds->findAndGetUint16(DCM_Rows, rows);
    ds->findAndGetUint16(DCM_Columns, cols);
    ds->findAndGetUint16(DCM_BitsAllocated, bitsAlloc);
    ds->findAndGetUint16(DCM_SamplesPerPixel, spp);
    if (!rows || !cols) {
        if (why) *why = "Missing Rows/Columns";
        std::cerr << "[ERR][DICOM] Missing Rows/Columns\n";
        return false;
    }
    if (!bitsAlloc) bitsAlloc = 16;
    if (!spp)       spp       = 1;

    // NumberOfFrames
    uint32_t nFrames = 1;
    {
        OFString nf;
        if (ds->findAndGetOFString(DCM_NumberOfFrames, nf).good()) {
            try { nFrames = std::max(1, std::stoi(nf.c_str())); } catch(...) {}
        }
    }

    std::cerr << "[DBG][DICOM][MIN16] rows=" << rows
              << " cols=" << cols
              << " BitsAllocated=" << bitsAlloc
              << " SamplesPerPixel=" << spp
              << " NumberOfFrames=" << nFrames
              << "\n";

    // PixelData
    DcmElement* pixelElem = nullptr;
    st = ds->findAndGetElement(DCM_PixelData, pixelElem, /*searchIntoSub=*/true);
    if (st.bad() || !pixelElem) {
        if (why) *why = "No PixelData";
        std::cerr << "[ERR][DICOM] Cannot find PixelData: " << st.text() << "\n";
        return false;
    }

    const uint32_t framePixels = static_cast<uint32_t>(rows) * static_cast<uint32_t>(cols);
    frames16.clear();

    if (spp != 1) {
        if (why) *why = "Only mono supported for 16-bit";
        std::cerr << "[ERR][DICOM][MIN16] SamplesPerPixel=" << spp << " not supported\n";
        return false;
    }

    // If the file is 8-bit, up-convert (clarity over fidelity)
    if (bitsAlloc == 8) {
        std::vector<cv::Mat> frames8;
        if (!dcmtk_read_one_file_u8(path, frames8, why)) return false;
        frames16.reserve(frames8.size());
        for (const auto& m8 : frames8) {
            cv::Mat m16; m8.convertTo(m16, CV_16U, 257.0);
            frames16.emplace_back(std::move(m16));
        }
        std::cerr << "[DBG][DICOM][MIN16] up-converted " << frames16.size() << " frame(s) 8→16 bit\n";
        return !frames16.empty();
    }

    // 16-bit mono: direct copy per frame
    Uint16* base16 = nullptr;
    st = pixelElem->getUint16Array(base16);
    if (st.bad() || !base16) {
        if (why) *why = "getUint16Array failed (maybe compressed?)";
        std::cerr << "[ERR][DICOM] getUint16Array failed: " << st.text() << "\n";
        return false;
    }
    const Uint32 totalWords = pixelElem->getLength() / 2;
    const Uint64 expected   = static_cast<Uint64>(framePixels) * static_cast<Uint64>(nFrames);
    if (totalWords < expected) {
        std::cerr << "[WRN][DICOM][MIN16] 16-bit PixelData smaller than expected ("
                  << totalWords << " < " << expected << "); forcing nFrames=1\n";
        nFrames = 1;
    }

    frames16.reserve(nFrames);
    for (uint32_t f = 0; f < nFrames; ++f) {
        const Uint16* src = base16 + static_cast<size_t>(f) * framePixels;
        cv::Mat m(rows, cols, CV_16UC1);
        std::memcpy(m.data, src, framePixels * sizeof(uint16_t));
        frames16.emplace_back(std::move(m));
        std::cerr << "[DBG][DICOM][MIN16] produced frame " << (f+1) << "/" << nFrames << "\n";
    }

#if DCMTK_SKIP_FF_DESTRUCTOR_ON_SUCCESS
    ff.release();
#endif
    return true;
}

// ------------------------------
// Series collector helpers
// ------------------------------
struct SeriesItem { std::string path; int inst; double z; };

static void sort_series_items(std::vector<SeriesItem>& items) {
    std::sort(items.begin(), items.end(), [](const SeriesItem& a, const SeriesItem& b){
        if (a.inst != b.inst) return a.inst < b.inst;
        return a.z < b.z;
    });
}

// Build a whole series stack (8-bit) from a single-seed file
static bool collect_series_from_seed_file_u8(const std::string& seed,
                                             std::vector<cv::Mat>& frames,
                                             std::string* why)
{
    std::error_code ec;
    fs::path p(seed);

    if (!fs::is_regular_file(p, ec)) {
        if (why) *why = "Seed is not a regular file";
        std::cerr << "[ERR][DICOM][SERIES] seed is not a regular file\n";
        return false;
    }

    std::string uid; int inst = 0; double z = 0.0;
    if (!dcmtk_series_key(seed, uid, inst, z)) {
        std::cerr << "[DBG][DICOM][SERIES] seed has no UID; fallback to single decode\n";
        return dcmtk_read_one_file_u8(seed, frames, why);
    }

    fs::path dir = p.parent_path();
    std::cerr << "[DBG][DICOM][SERIES] seed='" << seed << "' UID='" << uid
              << "' dir='" << dir.string() << "'\n";

    std::vector<SeriesItem> items;
    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file(ec)) continue;
        const auto ext = to_lower_copy(e.path().extension().string());
        if (!(ext.empty() || ext == ".dcm" || ext == ".dicom" || ext == ".ima")) continue;

        std::string uid2; int inst2 = 0; double z2 = 0.0;
        if (!dcmtk_series_key(e.path().string(), uid2, inst2, z2)) continue;
        if (uid2 == uid) items.push_back({ e.path().string(), inst2, z2 });
    }

    sort_series_items(items);
    std::cerr << "[DBG][DICOM][SERIES] series items=" << items.size() << "\n";

    frames.clear();
    size_t appended = 0;
    for (const auto& it : items) {
        std::vector<cv::Mat> fr;
        if (!dcmtk_read_one_file_u8(it.path, fr, why)) {
            std::cerr << "[WRN][DICOM][SERIES] skip unreadable: " << it.path << "\n";
            continue;
        }
        for (auto& m : fr) { frames.push_back(std::move(m)); ++appended; }
        std::cerr << "[DBG][DICOM][SERIES] appended " << fr.size()
                  << " from " << it.path
                  << " (inst=" << it.inst << ", z=" << it.z << ")\n";
    }
    std::cerr << "[DBG][DICOM][SERIES] total frames=" << frames.size()
              << " (appended=" << appended << ")\n";
    return !frames.empty();
}

// Build a whole series stack (16-bit) from a single-seed file
static bool collect_series_from_seed_file_u16(const std::string& seed,
                                              std::vector<cv::Mat>& frames16,
                                              std::string* why)
{
    std::error_code ec;
    fs::path p(seed);

    if (!fs::is_regular_file(p, ec)) {
        if (why) *why = "Seed is not a regular file";
        std::cerr << "[ERR][DICOM][SERIES] seed is not a regular file\n";
        return false;
    }

    std::string uid; int inst = 0; double z = 0.0;
    if (!dcmtk_series_key(seed, uid, inst, z)) {
        std::cerr << "[DBG][DICOM][SERIES] seed has no UID; fallback to single decode (16-bit)\n";
        return dcmtk_read_one_file_u16(seed, frames16, why);
    }

    fs::path dir = p.parent_path();
    std::cerr << "[DBG][DICOM][SERIES] (16-bit) seed='" << seed << "' UID='" << uid
              << "' dir='" << dir.string() << "'\n";

    std::vector<SeriesItem> items;
    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file(ec)) continue;
        const auto ext = to_lower_copy(e.path().extension().string());
        if (!(ext.empty() || ext == ".dcm" || ext == ".dicom" || ext == ".ima")) continue;

        std::string uid2; int inst2 = 0; double z2 = 0.0;
        if (!dcmtk_series_key(e.path().string(), uid2, inst2, z2)) continue;
        if (uid2 == uid) items.push_back({ e.path().string(), inst2, z2 });
    }

    sort_series_items(items);
    std::cerr << "[DBG][DICOM][SERIES] (16-bit) series items=" << items.size() << "\n";

    frames16.clear();
    size_t appended = 0;
    for (const auto& it : items) {
        std::vector<cv::Mat> fr16;
        if (!dcmtk_read_one_file_u16(it.path, fr16, why)) {
            std::cerr << "[WRN][DICOM][SERIES] (16-bit) skip unreadable: " << it.path << "\n";
            continue;
        }
        for (auto& m : fr16) { frames16.push_back(std::move(m)); ++appended; }
        std::cerr << "[DBG][DICOM][SERIES] (16-bit) appended " << fr16.size()
                  << " from " << it.path
                  << " (inst=" << it.inst << ", z=" << it.z << ")\n";
    }
    std::cerr << "[DBG][DICOM][SERIES] (16-bit) total frames=" << frames16.size()
              << " (appended=" << appended << ")\n";
    return !frames16.empty();
}

} // anon namespace

// ========================================================
// Exported functions (namespace io)
// ========================================================
namespace io {

bool read_dicom_frames_gray8(const std::string& pathOrDir,
                             std::vector<cv::Mat>& frames,
                             std::string* why)
{
    using std::cerr;
    using std::endl;

    std::error_code ec;
    const fs::path p(pathOrDir);
    frames.clear();

    // ------------------------------
    // SINGLE FILE PATH
    // ------------------------------
    if (fs::is_regular_file(p, ec)) {
        cerr << "[DBG][DICOM][IO] FILE (gray8): " << pathOrDir << endl;

        // 1) Decode the seed file first (handles multi-frame Enhanced DICOM)
        std::vector<cv::Mat> seedFrames;
        bool okDecode = false;
        try {
            okDecode = dcmtk_read_one_file_u8(pathOrDir, seedFrames, why);
            cerr << "[DBG][DICOM][IO] seed decode ok=" << okDecode
                 << " frames=" << seedFrames.size() << endl;
        } catch (const std::exception& e) {
            if (why) *why = e.what();
            cerr << "[DBG][DICOM][IO][EXC] seed decode exception: " << e.what() << endl;
            return false;
        } catch (...) {
            if (why) *why = "unknown exception in seed decode";
            cerr << "[DBG][DICOM][IO][EXC] unknown in seed decode" << endl;
            return false;
        }

        if (!okDecode) {
            cerr << "[DBG][DICOM][IO] seed decode failed; aborting\n";
            return false;
        }

        // Multi-frame seed? great—use it and we’re done.
        if (seedFrames.size() >= 2) {
            frames.swap(seedFrames);
            cerr << "[DBG][DICOM][IO] multi-frame seed -> using " << frames.size() << " frames\n";
            return true;
        }

        // 2) Single-frame seed → try building a series from folder (same SeriesInstanceUID)
        cerr << "[DBG][DICOM][IO] seed has 1 frame; trying folder series collection…\n";

        // Extract seed UID with helper (ensures consistency)
        std::string targetUID; int dummyInst = 0; double dummyZ = 0.0;
        if (!dcmtk_series_key(pathOrDir, targetUID, dummyInst, dummyZ) || targetUID.empty()) {
            cerr << "[DBG][DICOM][SERIES] seed has no SeriesInstanceUID; using seed frame only\n";
            frames.swap(seedFrames);
            return !frames.empty();
        }

        const fs::path dir = p.parent_path();
        struct Item { std::string path; int inst; double z; };
        std::vector<Item> items;

        for (auto& e : fs::directory_iterator(dir)) {
            if (!e.is_regular_file(ec)) continue;
            const auto ext = to_lower_copy(e.path().extension().string());
            if (!(ext.empty() || ext == ".dcm" || ext == ".dicom" || ext == ".ima")) continue;

            std::string uid2; int inst2 = 0; double z2 = 0.0;
            if (!dcmtk_series_key(e.path().string(), uid2, inst2, z2)) continue;
            if (uid2 != targetUID) continue;

            items.push_back({ e.path().string(), inst2, z2 });
        }

        std::sort(items.begin(), items.end(), [](const Item& a, const Item& b){
            if (a.inst != b.inst) return a.inst < b.inst;
            return a.z < b.z;
        });
        cerr << "[DBG][DICOM][SERIES] series items=" << items.size()
             << " (UID=" << targetUID << ")\n";

        // Decode each file and append all frames
        std::vector<cv::Mat> seriesFrames;
        size_t appendedFiles = 0;
        for (const auto& it : items) {
            std::vector<cv::Mat> fr;
            if (!dcmtk_read_one_file_u8(it.path, fr, why)) {
                cerr << "[WRN][DICOM][SERIES] skip unreadable: " << it.path << "\n";
                continue;
            }
            for (auto& m : fr) seriesFrames.emplace_back(std::move(m));
            ++appendedFiles;
            cerr << "[DBG][DICOM][SERIES] appended " << fr.size()
                 << " from " << it.path
                 << " (inst=" << it.inst << ", z=" << it.z << ")\n";
        }
        cerr << "[DBG][DICOM][SERIES] total frames=" << seriesFrames.size()
             << " (files used=" << appendedFiles << ")\n";

        if (seriesFrames.size() >= 2) {
            frames.swap(seriesFrames);
            cerr << "[DBG][DICOM][IO] folder series -> using " << frames.size() << " frames\n";
            return true;
        }

        // Fall back to the single decoded frame
        frames.swap(seedFrames);
        cerr << "[DBG][DICOM][IO] no multi-slice series discovered; using single frame\n";
        return !frames.empty();
    }

    // ------------------------------
    // DIRECTORY PATH
    // ------------------------------
    if (!fs::is_directory(p, ec)) {
        if (why) *why = "Path is neither file nor directory";
        return false;
    }

    // Find a seed file in the directory
    std::string seed;
    for (auto& e : fs::directory_iterator(p)) {
        if (!e.is_regular_file(ec)) continue;
        const auto ext = to_lower_copy(e.path().extension().string());
        if (!(ext.empty() || ext == ".dcm" || ext == ".dicom" || ext == ".ima")) continue;
        seed = e.path().string();
        break;
    }
    if (seed.empty()) {
        if (why) *why = "No DICOM files in directory";
        return false;
    }

    // Determine target SeriesInstanceUID from seed (via helper)
    std::string targetUID; int seedInst = 0; double seedZ = 0.0;
    if (!dcmtk_series_key(seed, targetUID, seedInst, seedZ) || targetUID.empty()) {
        if (why) *why = "Cannot read SeriesInstanceUID from seed";
        return false;
    }
    cerr << "[DBG][DICOM][SERIES] seed UID=" << targetUID
         << " dir=" << p.string() << endl;

    struct Item { std::string path; int inst; double z; };
    std::vector<Item> items;

    for (auto& e : fs::directory_iterator(p)) {
        if (!e.is_regular_file(ec)) continue;
        const std::string f = e.path().string();
        const auto ext = to_lower_copy(e.path().extension().string());
        if (!(ext.empty() || ext == ".dcm" || ext == ".dicom" || ext == ".ima")) continue;

        std::string uid2; int inst=0; double z=0.0;
        if (!dcmtk_series_key(f, uid2, inst, z)) continue;
        if (uid2 != targetUID) continue;

        items.push_back({f, inst, z});
    }

    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b){
        if (a.inst != b.inst) return a.inst < b.inst;
        return a.z < b.z;
    });

    cerr << "[DBG][DICOM][SERIES] files in series: " << items.size() << endl;

    frames.clear();
    for (const auto& it : items) {
        std::vector<cv::Mat> fr;
        if (!dcmtk_read_one_file_u8(it.path, fr, why)) {
            std::cerr << "[WRN][DICOM][SERIES] skip unreadable: " << it.path << std::endl;
            continue;
        }
        for (auto& m : fr) frames.push_back(std::move(m));
        std::cerr << "[DBG][DICOM][SERIES] appended " << fr.size()
                  << " from " << it.path << std::endl;
    }
    std::cerr << "[DBG][DICOM][SERIES] total frames=" << frames.size() << std::endl;
    return !frames.empty();
}


// 16-bit frames API (declared in io.hpp)
// Returns CV_16UC1 frames. If the file is actually 8-bit, we up-convert (×257).
bool read_dicom_frames_u16(const std::string& path,
                           std::vector<cv::Mat>& out16,
                           std::string* why)
{
    using std::cerr;
    using std::endl;
    std::error_code ec;
    const fs::path p(path);

    if (fs::is_regular_file(p, ec)) {
        cerr << "[DBG][DICOM][IO] FILE (16-bit path): " << path << endl;
        bool ok = false;
        try {
            ok = collect_series_from_seed_file_u16(path, out16, why);
            cerr << "[DBG][DICOM][IO] FILE EXIT (16-bit) ok=" << ok
                 << " frames=" << out16.size() << endl;
        } catch (const std::exception& e) {
            if (why) *why = e.what();
            cerr << "[DBG][DICOM][IO][EXC] " << e.what() << endl;
            ok = false;
        } catch (...) {
            if (why) *why = "unknown exception";
            cerr << "[DBG][DICOM][IO][EXC] unknown" << endl;
            ok = false;
        }
        return ok;
    }

    if (!fs::is_directory(p, ec)) {
        if (why) *why = "Path is neither file nor directory";
        return false;
    }

    // Directory: collect by UID (16-bit path)
    std::string seed;
    for (auto& e : fs::directory_iterator(p)) {
        if (!e.is_regular_file(ec)) continue;
        const auto ext = to_lower_copy(e.path().extension().string());
        if (!(ext.empty() || ext == ".dcm" || ext == ".dicom" || ext == ".ima")) continue;
        seed = e.path().string();
        break;
    }
    if (seed.empty()) {
        if (why) *why = "No DICOM files in directory";
        return false;
    }

    std::string targetUID; int inst0=0; double z0=0.0;
    if (!dcmtk_series_key(seed, targetUID, inst0, z0) || targetUID.empty()) {
        if (why) *why = "Cannot read seed DICOM in directory";
        return false;
    }
    std::cerr << "[DBG][DICOM][SERIES] (16-bit) seed UID=" << targetUID
              << " dir=" << p.string() << std::endl;

    std::vector<SeriesItem> items;
    for (auto& e : fs::directory_iterator(p)) {
        if (!e.is_regular_file(ec)) continue;
        const std::string f = e.path().string();
        const auto ext = to_lower_copy(e.path().extension().string());
        if (!(ext.empty() || ext == ".dcm" || ext == ".dicom" || ext == ".ima")) continue;

        std::string uid2; int inst=0; double z=0.0;
        if (!dcmtk_series_key(f, uid2, inst, z)) continue;
        if (uid2 == targetUID) items.push_back({f, inst, z});
    }

    sort_series_items(items);
    std::cerr << "[DBG][DICOM][SERIES] (16-bit) files in series: " << items.size() << std::endl;

    out16.clear();
    for (const auto& it : items) {
        std::vector<cv::Mat> fr16;
        if (!dcmtk_read_one_file_u16(it.path, fr16, why)) {
            std::cerr << "[WRN][DICOM][SERIES] (16-bit) skip unreadable: " << it.path << std::endl;
            continue;
        }
        for (auto& m : fr16) out16.push_back(std::move(m));
        std::cerr << "[DBG][DICOM][SERIES] (16-bit) appended " << fr16.size()
                  << " from " << it.path << std::endl;
    }
    std::cerr << "[DBG][DICOM][SERIES] (16-bit) total frames=" << out16.size() << std::endl;
    return !out16.empty();
}

// ---------------------------------
// Minimal DICOM basic meta (only safe fields we know)
// ---------------------------------
bool read_dicom_basic_meta(const std::string& path, DicomMeta& out, std::string* why) {
    DcmFileFormat ff;
    OFCondition st = ff.loadFile(path.c_str());
    if (st.bad()) {
        if (why) *why = st.text();
        return false;
    }
    DcmDataset* ds = ff.getDataset();

    auto getS = [&](const DcmTagKey& k)->std::string {
        OFString s; if (ds->findAndGetOFString(k, s).good()) return std::string(s.c_str());
        return {};
    };

    // Populate a few common fields (match io.hpp)
    out.patientName = getS(DCM_PatientName);
    out.patientID   = getS(DCM_PatientID);
    out.studyDate   = getS(DCM_StudyDate);
    out.studyTime   = getS(DCM_StudyTime);
    out.softwareVersions = getS(DCM_SoftwareVersions);
    out.institutionName  = getS(DCM_InstitutionName);
    out.seriesDescription= getS(DCM_SeriesDescription);

    // Optional timing if present
    out.tr_ms = getS(DCM_RepetitionTime);
    out.te_ms = getS(DCM_EchoTime);
    out.ti_ms = getS(DCM_InversionTime);

    // Magnetic field strength (text)
    out.B0T = getS(DCM_MagneticFieldStrength);

    return true;
}

// ---------------------------------
// HDF5 — ISMRMRD XML
// ---------------------------------
bool read_hdf5_ismrmrd_meta(const std::string& path, DicomMeta& out, std::string* why) {
    try {
        H5::H5File f(path, H5F_ACC_RDONLY);
        if (!f.nameExists("/ismrmrd_xml")) {
            if (why) *why = "No /ismrmrd_xml in HDF5";
            return false;
        }
        H5::DataSet ds = f.openDataSet("/ismrmrd_xml");
        if (!ds.getDataType().isVariableStr()) {
            if (why) *why = "/ismrmrd_xml is not variable-length string";
            return false;
        }

        // Read vlen string
        char* cstr = nullptr;
        H5::StrType st = ds.getStrType();
        ds.read(&cstr, st);
        std::string xml(cstr ? cstr : "");

        // Reclaim vlen mem
        hid_t had = ds.getId();
        hid_t tid = st.getId();
        hid_t sid = ds.getSpace().getId();
        H5Dvlen_reclaim(tid, sid, H5P_DEFAULT, &cstr);

        // Minimal parse hook (extend as needed)
        pugi::xml_document doc;
        if (!doc.load_string(xml.c_str())) {
            if (why) *why = "Bad ISMRMRD XML";
            return false;
        }

        (void)out; // map fields as needed later
        return true;

    } catch (const H5::Exception& e) {
        if (why) *why = e.getDetailMsg();
        return false;
    } catch (const std::exception& e) {
        if (why) *why = e.what();
        return false;
    }
}

// ======== PNG writer ========
bool write_png(const std::string& outPath, const cv::Mat& img, std::string* why)
{
    std::cerr << "[IO][PNG] write_png ENTER path='" << outPath << "'\n";
    if (img.empty()) {
        if (why) *why = "Input image is empty";
        std::cerr << "[IO][PNG][ERR] empty image\n";
        return false;
    }

    // Make sure we have 8-bit 1ch or 3ch data for imwrite
    cv::Mat u8;
    if (img.type() == CV_8UC1 || img.type() == CV_8UC3) {
        u8 = img;
    } else if (img.type() == CV_16UC1) {
        // Scale 16-bit → 8-bit (clarity over perf)
        double mn, mx;
        cv::minMaxLoc(img, &mn, &mx);
        if (mx <= mn) mx = mn + 1.0;
        cv::Mat f32; img.convertTo(f32, CV_32F);
        f32 = (f32 - (float)mn) / (float)(mx - mn);
        f32.convertTo(u8, CV_8U, 255.0);
    } else if (img.channels() == 3 && img.depth() == CV_16U) {
        cv::Mat tmp; img.convertTo(tmp, CV_8U, 1.0/257.0); // simple downscale
        u8 = std::move(tmp);
    } else {
        // Fallback: convert anything else to 8-bit gray
        cv::Mat gray;
        if (img.channels() == 3) { cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); }
        else { img.convertTo(gray, CV_8U); }
        u8 = std::move(gray);
    }

    std::vector<int> params = { cv::IMWRITE_PNG_COMPRESSION, 3 };
    try {
        const bool ok = cv::imwrite(outPath, u8, params);
        std::cerr << "[IO][PNG] imwrite returned ok=" << ok
                  << " size=" << u8.cols << "x" << u8.rows
                  << " ch=" << u8.channels() << "\n";
        if (!ok && why) *why = "cv::imwrite returned false";
        return ok;
    } catch (const cv::Exception& e) {
        if (why) *why = e.what();
        std::cerr << "[IO][PNG][ERR] " << e.what() << "\n";
        return false;
    }
}

bool write_dicom_sc_gray8(const std::string& outPath, const cv::Mat& imgIn, std::string* why)
{
    std::cerr << "[IO][DICOM] write_dicom_sc_gray8 ENTER path='" << outPath << "'\n";
    if (imgIn.empty()) {
        if (why) *why = "Input image is empty";
        std::cerr << "[IO][DICOM][ERR] empty image\n";
        return false;
    }

    // Ensure 8-bit, single-channel (MONOCHROME2)
    cv::Mat u8;
    if (imgIn.type() == CV_8UC1) {
        u8 = imgIn;
    } else if (imgIn.type() == CV_16UC1) {
        double mn, mx;
        cv::minMaxLoc(imgIn, &mn, &mx);
        if (mx <= mn) mx = mn + 1.0;
        cv::Mat f32; imgIn.convertTo(f32, CV_32F);
        f32 = (f32 - (float)mn) / (float)(mx - mn);
        f32.convertTo(u8, CV_8U, 255.0);
    } else if (imgIn.channels() == 3) {
        cv::cvtColor(imgIn, u8, cv::COLOR_BGR2GRAY);
    } else {
        imgIn.convertTo(u8, CV_8U);
    }

    // DICOM dataset
    DcmFileFormat ff;
    DcmDataset* ds = ff.getDataset();

    // --- UIDs ---
    char sopInstanceUID[128] = {};
    // NOTE: Replace this root with your organization’s UID root if you have one.
    const char* kUID_ROOT = "1.2.826.0.1.3680043.2.1125.101"; // example root
    dcmGenerateUniqueIdentifier(sopInstanceUID, kUID_ROOT);

    // --- Minimal mandatory attributes for Secondary Capture ---
    ds->putAndInsertString(DCM_SOPClassUID, UID_SecondaryCaptureImageStorage);
    ds->putAndInsertString(DCM_SOPInstanceUID, sopInstanceUID);

    // Image Pixel module (grayscale 8-bit)
    const Uint16 rows = (Uint16)u8.rows;
    const Uint16 cols = (Uint16)u8.cols;
    ds->putAndInsertUint16(DCM_SamplesPerPixel, 1);
    ds->putAndInsertString(DCM_PhotometricInterpretation, "MONOCHROME2");
    ds->putAndInsertUint16(DCM_Rows, rows);
    ds->putAndInsertUint16(DCM_Columns, cols);
    ds->putAndInsertUint16(DCM_BitsAllocated, 8);
    ds->putAndInsertUint16(DCM_BitsStored,    8);
    ds->putAndInsertUint16(DCM_HighBit,       7);
    ds->putAndInsertUint16(DCM_PixelRepresentation, 0); // 0 = unsigned

    // Optional but helpful
    ds->putAndInsertString(DCM_Modality, "OT");            // Other
    ds->putAndInsertString(DCM_ConversionType, "WSD");     // Workstation

    // Pixel Data
    const unsigned long nbytes = (unsigned long)(rows) * (unsigned long)(cols);
    if (!u8.isContinuous()) {
        cv::Mat tmp = u8.clone();
        ds->putAndInsertUint8Array(DCM_PixelData,
                                   reinterpret_cast<const Uint8*>(tmp.data), nbytes);
    } else {
        ds->putAndInsertUint8Array(DCM_PixelData,
                                   reinterpret_cast<const Uint8*>(u8.data), nbytes);
    }

    // File Meta Information
    DcmMetaInfo* meta = ff.getMetaInfo();
    meta->putAndInsertString(DCM_MediaStorageSOPClassUID, UID_SecondaryCaptureImageStorage);
    meta->putAndInsertString(DCM_MediaStorageSOPInstanceUID, sopInstanceUID);
    meta->putAndInsertString(DCM_ImplementationClassUID, kUID_ROOT);

    // Save as Explicit VR Little Endian (uncompressed)
    const OFCondition st = ff.saveFile(outPath.c_str(), EXS_LittleEndianExplicit);
    if (st.bad()) {
        if (why) *why = st.text();
        std::cerr << "[IO][DICOM][ERR] saveFile failed: " << st.text() << "\n";
        return false;
    }

    std::cerr << "[IO][DICOM] write_dicom_sc_gray8 OK -> '" << outPath
              << "' size=" << cols << "x" << rows << "\n";
    return true;
}

} // namespace io
