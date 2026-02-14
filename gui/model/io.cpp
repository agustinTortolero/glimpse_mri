
#include "io.hpp"

#include <QDebug>
#include <QMap>
#include <QVariantMap>
#include <QString>
#include <QByteArray>
#include <cstdarg>
#include <cstdio>

#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcdeftag.h>



#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif


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



#ifndef DCMTK_SKIP_DI_DESTRUCTOR_ON_SUCCESS
#define DCMTK_SKIP_DI_DESTRUCTOR_ON_SUCCESS 0
#endif

#ifndef DCMTK_SKIP_FF_DESTRUCTOR_ON_SUCCESS
#define DCMTK_SKIP_FF_DESTRUCTOR_ON_SUCCESS 0
#endif


#ifndef DCMTK_FORCE_RAW_PIXEL_FALLBACK
#define DCMTK_FORCE_RAW_PIXEL_FALLBACK 0
#endif


#ifndef IO_DEBUG_DUMP_FIRST_PNG
#define IO_DEBUG_DUMP_FIRST_PNG 1
#endif


#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


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
#include <cmath>

namespace fs = std::filesystem;


#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmdata/dcuid.h>
#include <dcmtk/dcmdata/dcistrmf.h>
#include <dcmtk/dcmdata/dcostrmf.h>
#include <dcmtk/dcmdata/dcrledrg.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#include <dcmtk/dcmjpeg/djdecode.h>
#include <dcmtk/dcmjpls/djdecode.h>
#ifdef WITH_DCMTK_DCMJ2K
#include <dcmtk/dcmj2k/djdecode.h>
#endif


#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcuid.h>
#include <dcmtk/dcmdata/dcostrmb.h>
#include <dcmtk/dcmdata/dcmetinf.h>


#include <H5Cpp.h>
#include <hdf5.h>
#include <pugixml.hpp>




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

    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(128, std::ios::beg);
    char tag[4] = {0,0,0,0};
    f.read(tag, 4);
    return (f.gcount() == 4 && tag[0]=='D' && tag[1]=='I' && tag[2]=='C' && tag[3]=='M');
}
static bool is_hdf5_magic(const std::string& path) {

    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    unsigned char sig[8] = {};
    f.read(reinterpret_cast<char*>(sig), 8);
    if (f.gcount() != 8) return false;
    const unsigned char ref[8] = { 0x89, 'H', 'D', 'F', 0x0d, 0x0a, 0x1a, 0x0a };
    for (int i = 0; i < 8; ++i) if (sig[i] != ref[i]) return false;
    return true;
}




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


    dbg_line(dbg, "[IO][probe] -> Unknown (treat as HDF5_Unknown for UI)");
    return {Flavor::HDF5_Unknown, ""};
}

}




namespace {








static bool dcmtk_series_key(const std::string& path,
                             std::string& uid,
                             int&         instanceNumber,
                             double&      zproj)
{
    std::cerr << "[DBG][DICOM][SERIES_KEY] ENTER path=" << path << "\n";


    io::dcmtk_global_init();

    DcmFileFormat ff;


    const E_TransferSyntax readXfer = EXS_Unknown;
    const E_GrpLenEncoding glenc    = EGL_noChange;
    const Uint32           maxRead  = 64u * 1024u;
    const E_FileReadMode   mode     = ERM_autoDetect;

    OFCondition st = ff.loadFile(path.c_str(), readXfer, glenc, maxRead, mode);
    if (st.bad()) {
        std::cerr << "[DBG][DICOM][SERIES_KEY] loadFile failed: " << st.text() << "\n";
        return false;
    }
    DcmDataset* ds = ff.getDataset();
    if (!ds) {
        std::cerr << "[DBG][DICOM][SERIES_KEY] no dataset\n";
        return false;
    }


    OFString sUID;
    if (ds->findAndGetOFString(DCM_SeriesInstanceUID, sUID).good() && !sUID.empty()) {
        uid = sUID.c_str();
    } else {
        std::cerr << "[DBG][DICOM][SERIES_KEY] missing SeriesInstanceUID\n";
        return false;
    }


    Sint32 instTmp = -1;
    if (ds->findAndGetSint32(DCM_InstanceNumber, instTmp).good()) {
        instanceNumber = static_cast<int>(instTmp);
    } else {

        instanceNumber = -1;
        std::cerr << "[DBG][DICOM][SERIES_KEY] InstanceNumber not present\n";
    }



    zproj = 0.0;
    OFString ippStr;
    if (ds->findAndGetOFStringArray(DCM_ImagePositionPatient, ippStr).good() && !ippStr.empty()) {


        std::string s = ippStr.c_str();
        double xyz[3] = {0,0,0};
        int idx = 0;
        std::string cur;
        for (char c : s) {
            if (c == '\\') {
                if (idx < 3) {
                    try { xyz[idx] = std::stod(cur); } catch (...) {}
                }
                cur.clear();
                ++idx;
            } else {
                cur.push_back(c);
            }
        }
        if (idx <= 3) {
            if (idx < 3) {

                try { xyz[idx] = std::stod(cur); } catch (...) {}
            } else {

            }
        }
        zproj = xyz[2];
        std::cerr << "[DBG][DICOM][SERIES_KEY] z from ImagePositionPatient = " << zproj << "\n";
    } else {

        double sl = 0.0;
        if (ds->findAndGetFloat64(DCM_SliceLocation, sl).good()) {
            zproj = sl;
            std::cerr << "[DBG][DICOM][SERIES_KEY] z from SliceLocation = " << zproj << "\n";
        } else {

            std::cerr << "[DBG][DICOM][SERIES_KEY] no z info; default 0.0\n";
            zproj = 0.0;
        }
    }

    std::cerr << "[DBG][DICOM][SERIES_KEY] EXIT uid=" << uid
              << " inst=" << instanceNumber << " z=" << zproj << "\n";
    return true;
}






static bool dcmtk_read_one_file_u8(const std::string& path,
                                   std::vector<cv::Mat>& frames,
                                   std::string* why)
{
    std::cerr << "[DBG][DICOM][DCMTK] ENTER read_u8: " << path << "\n";
    io::dcmtk_global_init();
    frames.clear();

#if !DCMTK_FORCE_RAW_PIXEL_FALLBACK

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


            di->setMinMaxWindow();

            frames.reserve(nFrames);
            for (unsigned long f = 0; f < nFrames; ++f) {
                const void* out = di->getOutputData(8, f, 0 );
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


    uint32_t nFrames = 1;
    {
        OFString nf;
        if (ds->findAndGetOFString(DCM_NumberOfFrames, nf).good()) {
            try {
                int tmp = std::max(1, std::stoi(nf.c_str()));
                nFrames = static_cast<uint32_t>(tmp);
            } catch(...) {  }
        }
    }


    Uint16 pixRep = 0;
    ds->findAndGetUint16(DCM_PixelRepresentation, pixRep);

    std::cerr << "[DBG][DICOM][MIN] rows=" << rows
              << " cols=" << cols
              << " BitsAllocated=" << bitsAlloc
              << " SamplesPerPixel=" << spp
              << " NumberOfFrames=" << nFrames
              << " PixelRepresentation=" << pixRep
              << "\n";


    DcmElement* elem = nullptr;
    st = ds->findAndGetElement(DCM_PixelData, elem, true);
    if (st.bad() || !elem) {
        if (why) *why = "No PixelData";
        std::cerr << "[ERR][DICOM] Cannot find PixelData: " << st.text() << "\n";
        return false;
    }

    const uint32_t framePixels = static_cast<uint32_t>(rows) * static_cast<uint32_t>(cols);
    frames.clear();


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


    DcmElement* pixelElem = nullptr;
    st = ds->findAndGetElement(DCM_PixelData, pixelElem, true);
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




struct SeriesItem { std::string path; int inst; double z; };

static void sort_series_items(std::vector<SeriesItem>& items) {
    std::sort(items.begin(), items.end(), [](const SeriesItem& a, const SeriesItem& b){
        if (a.inst != b.inst) return a.inst < b.inst;
        return a.z < b.z;
    });
}


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

}




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




    if (fs::is_regular_file(p, ec)) {
        cerr << "[DBG][DICOM][IO] FILE (gray8): " << pathOrDir << endl;


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


        if (seedFrames.size() >= 2) {
            frames.swap(seedFrames);
            cerr << "[DBG][DICOM][IO] multi-frame seed -> using " << frames.size() << " frames\n";
            return true;
        }


        cerr << "[DBG][DICOM][IO] seed has 1 frame; trying folder series collection…\n";


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


        frames.swap(seedFrames);
        cerr << "[DBG][DICOM][IO] no multi-slice series discovered; using single frame\n";
        return !frames.empty();
    }




    if (!fs::is_directory(p, ec)) {
        if (why) *why = "Path is neither file nor directory";
        return false;
    }


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


bool read_dicom_frames_u16(const std::string& path,
                               std::vector<cv::Mat>& out16,
                               std::string* why)
{
    out16.clear();
    if (why) why->clear();

    std::error_code ec;
    fs::path p(path);

    if (!fs::exists(p, ec)) {
        if (why) *why = "Path does not exist: " + path;
        std::cerr << "[ERR][DICOM][IO] path not found: " << path << "\n";
        return false;
    }

    auto decode_one_file_u16 = [&](const std::string& filePath) -> bool {
        std::cerr << "[DBG][DICOM][IO] decode FILE (16-bit path): " << filePath << "\n";
        io::dcmtk_global_init();

        DicomImage di(filePath.c_str());
        if (di.getStatus() != EIS_Normal) {
            if (why) *why = "DicomImage status != normal";
            std::cerr << "[ERR][DICOM][IO] DicomImage status != normal\n";
            return false;
        }

        const int frames = static_cast<int>(di.getFrameCount());
        const int rows   = static_cast<int>(di.getHeight());
        const int cols   = static_cast<int>(di.getWidth());
        if (rows <= 0 || cols <= 0) {
            if (why) *why = "Invalid rows/cols";
            std::cerr << "[ERR][DICOM][IO] invalid rows/cols: " << rows << "x" << cols << "\n";
            return false;
        }

        std::cerr << "[DBG][DICOM][IO] dims=" << cols << "x" << rows
                  << " frames=" << frames << "\n";


        for (int f = 0; f < std::max(frames, 1); ++f) {

            std::vector<uint16_t> buf16(static_cast<size_t>(rows) * static_cast<size_t>(cols));
            bool ok16 = di.getOutputData(buf16.data(),
                                         static_cast<unsigned long>(buf16.size() * sizeof(uint16_t)),
                                         f, 16);
            if (!ok16) {
                std::cerr << "[WRN][DICOM][IO] getOutputData(16) failed on frame " << f
                          << " -> trying 8-bit fallback\n";

                std::vector<uint8_t> buf8(static_cast<size_t>(rows) * static_cast<size_t>(cols));
                bool ok8 = di.getOutputData(buf8.data(),
                                            static_cast<unsigned long>(buf8.size()),
                                            f, 8);
                if (!ok8) {
                    if (why) *why = "Neither 16b nor 8b output available";
                    std::cerr << "[ERR][DICOM][IO] getOutputData(8) also failed on frame " << f << "\n";
                    return false;
                }


                for (size_t i = 0; i < buf16.size(); ++i) {
                    buf16[i] = static_cast<uint16_t>(buf8[i]) << 8;
                }
            }


            cv::Mat frame(rows, cols, CV_16UC1);
            std::memcpy(frame.data, buf16.data(), buf16.size() * sizeof(uint16_t));
            out16.emplace_back(std::move(frame));

            std::cerr << "[DBG][DICOM][IO] frame " << f << " appended (16-bit)\n";
        }

        std::cerr << "[DBG][DICOM][IO] FILE EXIT ok=1 frames_appended=" << out16.size() << "\n";
        return true;
    };

    if (fs::is_regular_file(p, ec)) {

        const bool ok = decode_one_file_u16(path);
        if (!ok && why && why->empty()) *why = "Failed to decode DICOM file";
        return ok;
    }

    if (fs::is_directory(p, ec)) {
        std::cerr << "[DBG][DICOM][IO] DIR (16-bit path): " << path << "\n";

        struct Rec {
            std::string uid;
            double      z;
            int         inst;
            std::string file;
        };
        std::vector<Rec> recs;


        size_t probed = 0, admitted = 0;
        for (const auto& de : fs::directory_iterator(p, ec)) {
            if (ec) break;
            if (!de.is_regular_file()) continue;

            const std::string fpath = de.path().string();
            ++probed;

            std::string uid;
            int inst = -1;
            double z  = 0.0;
            if (!dcmtk_series_key(fpath, uid, inst, z)) {
                continue;
            }
            recs.push_back({uid, z, inst, fpath});
            ++admitted;
        }
        std::cerr << "[DBG][DICOM][IO] probed=" << probed << " admitted=" << admitted << "\n";
        if (recs.empty()) {
            if (why) *why = "No DICOM files admitted in directory";
            return false;
        }


        std::unordered_map<std::string, int> counts;
        for (auto& r : recs) counts[r.uid]++;

        std::string bestUID;
        int bestCount = -1;
        for (auto& kv : counts) {
            if (kv.second > bestCount) { bestCount = kv.second; bestUID = kv.first; }
        }
        std::cerr << "[DBG][DICOM][IO] best series uid=" << bestUID << " count=" << bestCount << "\n";


        std::vector<Rec> seq;
        seq.reserve(static_cast<size_t>(bestCount));
        for (auto& r : recs) if (r.uid == bestUID) seq.push_back(r);

        std::sort(seq.begin(), seq.end(), [](const Rec& a, const Rec& b){
            if (a.z != b.z)     return a.z < b.z;
            if (a.inst != b.inst) return a.inst < b.inst;
            return a.file < b.file;
        });

        std::cerr << "[DBG][DICOM][IO] decoding " << seq.size() << " file(s) for best series…\n";


        size_t before = out16.size();
        for (size_t i = 0; i < seq.size(); ++i) {
            std::cerr << "[DBG][DICOM][IO] [" << i+1 << "/" << seq.size() << "] " << seq[i].file << "\n";
            if (!decode_one_file_u16(seq[i].file)) {
                std::cerr << "[WRN][DICOM][IO] skipping file due to decode error\n";
            }
        }
        size_t appended = out16.size() - before;
        std::cerr << "[DBG][DICOM][IO] DIR EXIT frames_appended=" << appended << "\n";
        return appended > 0;
    }

    if (why) *why = "Path is neither file nor directory";
    std::cerr << "[ERR][DICOM][IO] not a file/dir: " << path << "\n";
    return false;
}


bool read_dicom_basic_meta(const std::string& path, DicomMeta& out, std::string* why)
{

    auto setWhy = [&](const char* s){ if (why) *why = s ? s : ""; };

    auto dbg = [&](const char* fmt, ...) {
        va_list ap; va_start(ap, fmt);
        std::fprintf(stderr, "[DBG][DICOM][META] ");
        std::vfprintf(stderr, fmt, ap);
        std::fprintf(stderr, "\n");
        va_end(ap);
    };

    auto warn = [&](const char* fmt, ...) {
        va_list ap; va_start(ap, fmt);
        std::fprintf(stderr, "[WRN][DICOM][META] ");
        std::vfprintf(stderr, fmt, ap);
        std::fprintf(stderr, "\n");
        va_end(ap);
    };

    auto errp = [&](const char* fmt, ...) {
        va_list ap; va_start(ap, fmt);
        std::fprintf(stderr, "[ERR][DICOM][META] ");
        std::vfprintf(stderr, fmt, ap);
        std::fprintf(stderr, "\n");
        va_end(ap);
    };

    auto trim_ascii_inplace = [](std::string& s) {
        auto is_ws = [](unsigned char c){ return c <= 0x20 || c == 0x7F; };
        size_t a = 0, b = s.size();
        while (a < b && is_ws(static_cast<unsigned char>(s[a]))) ++a;
        while (b > a && is_ws(static_cast<unsigned char>(s[b-1]))) --b;
        if (a != 0 || b != s.size()) s = s.substr(a, b - a);
    };

    auto first_multivalue = [](const std::string& s) {
        const auto p = s.find('\\');
        return (p == std::string::npos) ? s : s.substr(0, p);
    };

    auto normalize_dicom_time = [](std::string tm) {
        if (tm.empty()) return tm;
        std::string frac;
        const auto dot = tm.find('.');
        if (dot != std::string::npos) { frac = tm.substr(dot); tm = tm.substr(0, dot); }
        while (tm.size() < 6) tm.push_back('0');
        tm = tm.substr(0,2) + ":" + tm.substr(2,2) + ":" + tm.substr(4,2) + frac;
        return tm;
    };

    auto is_valid_utf8 = [](const char* s, size_t n) -> bool {
        size_t i = 0;
        while (i < n) {
            unsigned char c = static_cast<unsigned char>(s[i]);
            if (c < 0x80) { ++i; continue; }
            size_t need = (c & 0xE0) == 0xC0 ? 1 :
                              (c & 0xF0) == 0xE0 ? 2 :
                              (c & 0xF8) == 0xF0 ? 3 : 0;
            if (!need || i + need >= n) return false;
            if ((c & 0xF8) == 0xF8 || (c & 0xC0) == 0x80) return false;
            for (size_t k = 1; k <= need; ++k) {
                unsigned char cc = static_cast<unsigned char>(s[i+k]);
                if ((cc & 0xC0) != 0x80) return false;
            }
            i += need + 1;
        }
        return true;
    };

    auto latin1_to_utf8 = [](const char* s, size_t n) -> std::string {
        std::string out;
        out.reserve(n * 2);
        for (size_t i = 0; i < n; ++i) {
            unsigned char c = static_cast<unsigned char>(s[i]);
            if (c < 0x80) {
                out.push_back(static_cast<char>(c));
            } else {
                out.push_back(static_cast<char>(0xC0 | (c >> 6)));
                out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
            }
        }
        return out;
    };

    auto printable_preview = [&](const char* p, size_t n, size_t max_chars = 128) -> std::string {
        std::string out;
        if (!p || n == 0) return out;
        size_t m = (n < max_chars) ? n : max_chars;
        out.reserve(m);
        for (size_t i = 0; i < m; ++i) {
            unsigned char c = static_cast<unsigned char>(p[i]);
            if (c >= 0x20 && c < 0x7F) out.push_back(static_cast<char>(c));
            else                       out.push_back('.');
        }
        if (m < n) out += "...";
        return out;
    };

    auto is_text_vr = [](DcmVR vr) -> bool {
        switch (vr.getEVR()) {
        case EVR_AE: case EVR_AS: case EVR_CS: case EVR_DA:
        case EVR_DS: case EVR_DT: case EVR_IS: case EVR_LO:
        case EVR_LT: case EVR_PN: case EVR_SH: case EVR_ST:
        case EVR_TM: case EVR_UI: case EVR_UT:
#ifdef EVR_UC
        case EVR_UC:
#endif
            return true;
        default: return false;
        }
    };


    auto fetch_text_utf8 = [&](DcmDataset* ds, const DcmTagKey& key, const char* name) -> std::string {
        if (!ds) return {};
        DcmElement* el = nullptr;
        OFCondition st = ds->findAndGetElement(key, el);
        if (st.bad() || !el) {
            warn("%s: element not found", name ? name : "TAG");
            return {};
        }
        const DcmTag& tag = el->getTag();
        char tagStr[32];
        std::snprintf(tagStr, sizeof(tagStr), "(%04X,%04X)",
                      (unsigned)tag.getGroup(), (unsigned)tag.getElement());
        DcmVR vr = el->getVR();
        const char* vrName = vr.getVRName(); if (!vrName) vrName = "??";
        const unsigned long vlen = el->getLength();

        dbg("%s: tag=%s VR=%s len=%lu", name ? name : "TAG", tagStr, vrName, (unsigned long)vlen);

        if (!is_text_vr(vr)) {
            warn("%s: non-text VR; returning empty", name ? name : "TAG");
            return {};
        }

        const char* p = nullptr;
        st = ds->findAndGetString(key, p );
        if (st.bad() || !p) {
            dbg("%s: findAndGetString => %s; returning empty", name ? name : "TAG",
                st.good() ? "null" : st.text());
            return {};
        }


        size_t n = std::strlen(p);
        const size_t CAP = 1u << 20;
        if (n > CAP) { warn("%s: value too large (%zu), capping to %zu", name, n, CAP); n = CAP; }

        std::string s;
        if (is_valid_utf8(p, n)) s.assign(p, p + n);
        else                     s = latin1_to_utf8(p, n);
        trim_ascii_inplace(s);
        return s;
    };

    auto trim_trailing_zeros = [](std::string s) {
        if (s.find('.') != std::string::npos) {
            while (!s.empty() && s.back() == '0') s.pop_back();
            if (!s.empty() && s.back() == '.')   s.pop_back();
        }
        return s;
    };

    auto get_number_as_string3 = [&](DcmDataset* ds, const DcmTagKey& key, const char* name) -> std::string {
        if (!ds) return {};
        Float64 f = 0.0;
        if (ds->findAndGetFloat64(key, f).good()) {
            char buf[64]; std::snprintf(buf, sizeof(buf), "%.3f", static_cast<double>(f));
            std::string r(buf); return trim_trailing_zeros(r);
        }
        std::string s = first_multivalue(fetch_text_utf8(ds, key, name));
        trim_ascii_inplace(s);
        return s;
    };


    std::fprintf(stderr, "[DBG][DICOM][META] ENTER path=%s\n", path.c_str());
    out = {};

    try {
        DcmFileFormat ff;
        const E_TransferSyntax readXfer = EXS_Unknown;
        const E_GrpLenEncoding glenc    = EGL_noChange;
        const Uint32           maxRead  = 64u * 1024u;
        const E_FileReadMode   mode     = ERM_autoDetect;

        dbg("loadFile(...) begin (cap=%u bytes/element)", (unsigned)maxRead);
        OFCondition st = ff.loadFile(path.c_str(), readXfer, glenc, maxRead, mode);
        if (st.bad()) {
            setWhy(st.text());
            errp("loadFile failed: %s", st.text());
            std::fprintf(stderr, "[DBG][DICOM][META] EXIT (fail)\n");
            return false;
        }
        dbg("loadFile(...) ok");

        DcmDataset* ds = ff.getDataset();
        if (!ds) {
            setWhy("No dataset");
            errp("null dataset");
            std::fprintf(stderr, "[DBG][DICOM][META] EXIT (fail)\n");
            return false;
        }


        dbg("TAG: Manufacturer...");
        out.manufacturer = fetch_text_utf8(ds, DCM_Manufacturer, "Manufacturer");
        dbg("  ok: '%s'", printable_preview(out.manufacturer.c_str(), out.manufacturer.size()).c_str());

        dbg("TAG: Model...");
        out.modelName = fetch_text_utf8(ds, DCM_ManufacturerModelName, "ModelName");
        dbg("  ok: '%s'", printable_preview(out.modelName.c_str(), out.modelName.size()).c_str());

        dbg("TAG: StudyDate...");
        out.studyDate = fetch_text_utf8(ds, DCM_StudyDate, "StudyDate");
        dbg("  ok: '%s'", printable_preview(out.studyDate.c_str(), out.studyDate.size()).c_str());

        dbg("TAG: StudyTime...");
        {
            std::string rawST = fetch_text_utf8(ds, DCM_StudyTime, "StudyTime");
            out.studyTime = normalize_dicom_time(first_multivalue(rawST));
            dbg("  ok: '%s'", printable_preview(out.studyTime.c_str(), out.studyTime.size()).c_str());
        }


        dbg("TAG: B0T...");
        out.B0T = get_number_as_string3(ds, DCM_MagneticFieldStrength, "MagneticFieldStrength");
        dbg("  ok: '%s'", printable_preview(out.B0T.c_str(), out.B0T.size()).c_str());

        dbg("TAG: TR...");
        out.tr_ms = get_number_as_string3(ds, DCM_RepetitionTime, "RepetitionTime");
        dbg("  ok: '%s'", printable_preview(out.tr_ms.c_str(), out.tr_ms.size()).c_str());

        dbg("TAG: TE...");
        out.te_ms = get_number_as_string3(ds, DCM_EchoTime, "EchoTime");
        dbg("  ok: '%s'", printable_preview(out.te_ms.c_str(), out.te_ms.size()).c_str());

        dbg("TAG: TI...");
        out.ti_ms = get_number_as_string3(ds, DCM_InversionTime, "InversionTime");
        dbg("  ok: '%s'", printable_preview(out.ti_ms.c_str(), out.ti_ms.size()).c_str());


        std::fprintf(stderr,
                     "[DBG][DICOM][META] OK Manuf='%s' Model='%s' Date='%s' Time='%s' "
                     "B0(T)='%s' TR(ms)='%s' TE(ms)='%s' TI(ms)='%s'\n",
                     printable_preview(out.manufacturer.c_str(), out.manufacturer.size(), 64).c_str(),
                     printable_preview(out.modelName.c_str(),    out.modelName.size(),    64).c_str(),
                     printable_preview(out.studyDate.c_str(),    out.studyDate.size(),    32).c_str(),
                     printable_preview(out.studyTime.c_str(),    out.studyTime.size(),    32).c_str(),
                     printable_preview(out.B0T.c_str(),          out.B0T.size(),          32).c_str(),
                     printable_preview(out.tr_ms.c_str(),        out.tr_ms.size(),        32).c_str(),
                     printable_preview(out.te_ms.c_str(),        out.te_ms.size(),        32).c_str(),
                     printable_preview(out.ti_ms.c_str(),        out.ti_ms.size(),        32).c_str());

        std::fprintf(stderr, "[DBG][DICOM][META] EXIT (ok)\n");
        return true;
    }
    catch (const std::exception& ex) {
        setWhy(ex.what());
        errp("EXC: %s", ex.what());
    }
    catch (...) {
        setWhy("Unknown exception in read_dicom_basic_meta");
        errp("EXC: unknown");
    }

    std::fprintf(stderr, "[DBG][DICOM][META] EXIT (fail)\n");
    return false;
}




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


        char* cstr = nullptr;
        H5::StrType st = ds.getStrType();
        ds.read(&cstr, st);
        std::string xml(cstr ? cstr : "");


        hid_t had = ds.getId();
        hid_t tid = st.getId();
        hid_t sid = ds.getSpace().getId();
        H5Dvlen_reclaim(tid, sid, H5P_DEFAULT, &cstr);


        pugi::xml_document doc;
        if (!doc.load_string(xml.c_str())) {
            if (why) *why = "Bad ISMRMRD XML";
            return false;
        }

        (void)out;
        return true;

    } catch (const H5::Exception& e) {
        if (why) *why = e.getDetailMsg();
        return false;
    } catch (const std::exception& e) {
        if (why) *why = e.what();
        return false;
    }
}


bool write_png(const std::string& outPath, const cv::Mat& img, std::string* why)
{
    std::cerr << "[IO][PNG] write_png ENTER path='" << outPath << "'\n";
    if (img.empty()) {
        if (why) *why = "Input image is empty";
        std::cerr << "[IO][PNG][ERR] empty image\n";
        return false;
    }


    cv::Mat u8;
    if (img.type() == CV_8UC1 || img.type() == CV_8UC3) {
        u8 = img;
    } else if (img.type() == CV_16UC1) {

        double mn, mx;
        cv::minMaxLoc(img, &mn, &mx);
        if (mx <= mn) mx = mn + 1.0;
        cv::Mat f32; img.convertTo(f32, CV_32F);
        f32 = (f32 - (float)mn) / (float)(mx - mn);
        f32.convertTo(u8, CV_8U, 255.0);
    } else if (img.channels() == 3 && img.depth() == CV_16U) {
        cv::Mat tmp; img.convertTo(tmp, CV_8U, 1.0/257.0);
        u8 = std::move(tmp);
    } else {

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


    DcmFileFormat ff;
    DcmDataset* ds = ff.getDataset();


    char sopInstanceUID[128] = {};

    const char* kUID_ROOT = "1.2.826.0.1.3680043.2.1125.101";
    dcmGenerateUniqueIdentifier(sopInstanceUID, kUID_ROOT);


    ds->putAndInsertString(DCM_SOPClassUID, UID_SecondaryCaptureImageStorage);
    ds->putAndInsertString(DCM_SOPInstanceUID, sopInstanceUID);


    const Uint16 rows = (Uint16)u8.rows;
    const Uint16 cols = (Uint16)u8.cols;
    ds->putAndInsertUint16(DCM_SamplesPerPixel, 1);
    ds->putAndInsertString(DCM_PhotometricInterpretation, "MONOCHROME2");
    ds->putAndInsertUint16(DCM_Rows, rows);
    ds->putAndInsertUint16(DCM_Columns, cols);
    ds->putAndInsertUint16(DCM_BitsAllocated, 8);
    ds->putAndInsertUint16(DCM_BitsStored,    8);
    ds->putAndInsertUint16(DCM_HighBit,       7);
    ds->putAndInsertUint16(DCM_PixelRepresentation, 0);


    ds->putAndInsertString(DCM_Modality, "OT");
    ds->putAndInsertString(DCM_ConversionType, "WSD");


    const unsigned long nbytes = (unsigned long)(rows) * (unsigned long)(cols);
    if (!u8.isContinuous()) {
        cv::Mat tmp = u8.clone();
        ds->putAndInsertUint8Array(DCM_PixelData,
                                   reinterpret_cast<const Uint8*>(tmp.data), nbytes);
    } else {
        ds->putAndInsertUint8Array(DCM_PixelData,
                                   reinterpret_cast<const Uint8*>(u8.data), nbytes);
    }


    DcmMetaInfo* meta = ff.getMetaInfo();
    meta->putAndInsertString(DCM_MediaStorageSOPClassUID, UID_SecondaryCaptureImageStorage);
    meta->putAndInsertString(DCM_MediaStorageSOPInstanceUID, sopInstanceUID);
    meta->putAndInsertString(DCM_ImplementationClassUID, kUID_ROOT);


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

bool logDicomMetaBrief(const QString& path)
{
    qDebug().noquote() << "[DBG][DICOM][META] ENTER path=" << path;

    DcmFileFormat file;
    const QByteArray pathBytes = path.toUtf8();
    const OFCondition st = file.loadFile(pathBytes.constData());
    if (st.bad()) {
        qWarning().noquote() << "[WRN][DICOM][META] loadFile failed:" << st.text();
        return false;
    }


    DcmDataset& ds = *file.getDataset();


    OFString csOF;
    (void)ds.findAndGetOFStringArray(DCM_SpecificCharacterSet, csOF);
    const QString csRaw = QString::fromLatin1(csOF.c_str(), static_cast<int>(csOF.size()));
    const QStringList csVals = csRaw.split('\\', Qt::SkipEmptyParts);


    auto decodeWithCS = [&](const OFString& s) -> QString {
        const QByteArray ba(s.c_str(), static_cast<int>(s.size()));
        auto tryUtf8   = [&](){ return QString::fromUtf8(ba);    };
        auto tryLatin1 = [&](){ return QString::fromLatin1(ba);  };

        bool preferLatin1 = false;
        bool preferUtf8   = false;
        for (const QString& v0 : csVals) {
            const QString v = v0.trimmed();
            if (v == QLatin1String("ISO_IR 192")) preferUtf8 = true;

            if (v == QLatin1String("ISO_IR 100") ||
                v == QLatin1String("ISO_IR 101") ||
                v == QLatin1String("ISO_IR 109") ||
                v == QLatin1String("ISO_IR 110") ||
                v == QLatin1String("ISO_IR 148"))
                preferLatin1 = true;
        }

        QString out = preferLatin1 ? tryLatin1()
                      : preferUtf8   ? tryUtf8()
                                   : tryUtf8();


        if (out.contains(QChar::ReplacementCharacter))
            out = tryLatin1();

        return out.trimmed();
    };

    auto getStr = [&](const DcmTagKey& key) -> QString {
        OFString v;
        if (ds.findAndGetOFStringArray(key, v).good())
            return decodeWithCS(v);
        return {};
    };

    auto getUInt = [&](const DcmTagKey& key, quint32& out) -> bool {

        Uint16 u16 = 0;
        if (ds.findAndGetUint16(key, u16).good()) { out = static_cast<quint32>(u16); return true; }
        Uint32 u32 = 0;
        if (ds.findAndGetUint32(key, u32).good()) { out = static_cast<quint32>(u32); return true; }
        Sint32 s32 = 0;
        if (ds.findAndGetSint32(key, s32).good() && s32 >= 0) { out = static_cast<quint32>(s32); return true; }

        const QString as = getStr(key);
        bool ok = false; const quint32 tmp = as.toUInt(&ok);
        if (ok) { out = tmp; return true; }
        return false;
    };

    auto logField = [&](const char* name, const QString& v) {
        qDebug().noquote() << "[DBG][DICOM][META]" << name << "="
                           << (v.isEmpty() ? "<empty>" : v);
    };

    auto logUInt = [&](const char* name, const DcmTagKey& key) {
        quint32 n = 0;
        if (getUInt(key, n))
            qDebug().noquote() << "[DBG][DICOM][META]" << name << "=" << n;
        else
            qDebug().noquote() << "[DBG][DICOM][META]" << name << "= <n/a>";
    };


    const QString specificCS   = csRaw;
    const QString manuf        = getStr(DCM_Manufacturer);
    const QString model        = getStr(DCM_ManufacturerModelName);
    const QString swVersions   = getStr(DCM_SoftwareVersions);

    const QString patientName  = getStr(DCM_PatientName);
    const QString patientID    = getStr(DCM_PatientID);
    const QString modality     = getStr(DCM_Modality);
    const QString studyDesc    = getStr(DCM_StudyDescription);
    const QString seriesDesc   = getStr(DCM_SeriesDescription);

    quint32 rows = 0, cols = 0, frames = 0;
    (void)getUInt(DCM_Rows, rows);
    (void)getUInt(DCM_Columns, cols);
    (void)getUInt(DCM_NumberOfFrames, frames);

    logField("SpecificCharacterSet(0008,0005)", specificCS);
    logField("Manufacturer",                    manuf);
    logField("ModelName",                       model);
    logField("SoftwareVersions",                swVersions);

    logField("PatientName",                     patientName);
    logField("PatientID",                       patientID);
    logField("Modality",                        modality);
    logField("StudyDescription",                studyDesc);
    logField("SeriesDescription",               seriesDesc);

    qDebug().noquote() << "[DBG][DICOM][META] Rows=" << rows
                       << " Columns=" << cols
                       << " Frames=" << (frames ? QString::number(frames) : QString("<n/a>"));

    qDebug().noquote() << "[QT][DBG][CTRL][loadDicom][META] EXIT";
    return true;
}


QVariantMap readDicomMetaMinimal(const QString& path)
{
    QVariantMap out;
    out.insert("path", path);

    DcmFileFormat file;
    const QByteArray pathBytes = path.toUtf8();
    const OFCondition st = file.loadFile(pathBytes.constData());
    if (st.bad()) {
        out.insert("error", QString::fromLatin1(st.text()));
        return out;
    }

    DcmDataset& ds = *file.getDataset();


    auto ofToQString = [](const OFString& s) -> QString {
        const QByteArray ba(s.c_str(), static_cast<int>(s.size()));
        QString out = QString::fromUtf8(ba);
        if (out.contains(QChar::ReplacementCharacter))
            out = QString::fromLatin1(ba);
        return out.trimmed();
    };

    auto getStr = [&](const DcmTagKey& key) -> QString {
        OFString v;
        if (ds.findAndGetOFStringArray(key, v).good())
            return ofToQString(v);
        return {};
    };

    auto getUInt = [&](const DcmTagKey& key) -> QVariant {
        Uint32 u = 0;
        if (ds.findAndGetUint32(key, u).good()) return QVariant::fromValue<quint32>(u);
        Sint32 s = 0;
        if (ds.findAndGetSint32(key, s).good() && s >= 0) return QVariant::fromValue<quint32>(s);
        const QString as = getStr(key);
        bool ok = false; const uint val = as.toUInt(&ok);
        return ok ? QVariant::fromValue<quint32>(val) : QVariant{};
    };


    out.insert("SpecificCharacterSet", getStr(DCM_SpecificCharacterSet));
    out.insert("Manufacturer",         getStr(DCM_Manufacturer));
    out.insert("ModelName",            getStr(DCM_ManufacturerModelName));
    out.insert("SoftwareVersions",     getStr(DCM_SoftwareVersions));

    out.insert("PatientName",          getStr(DCM_PatientName));
    out.insert("PatientID",            getStr(DCM_PatientID));
    out.insert("Modality",             getStr(DCM_Modality));
    out.insert("StudyDescription",     getStr(DCM_StudyDescription));
    out.insert("SeriesDescription",    getStr(DCM_SeriesDescription));

    out.insert("Rows",                 getUInt(DCM_Rows));
    out.insert("Columns",              getUInt(DCM_Columns));
    out.insert("NumberOfFrames",       getUInt(DCM_NumberOfFrames));

    return out;
}



}
