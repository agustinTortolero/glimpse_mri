// model/io.cpp
#include "io.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
#include <filesystem>

// HDF5
#pragma warning(push)
#pragma warning(disable:4251)
#include <H5public.h>
#include <H5Cpp.h>
#include <H5Epublic.h>
#pragma warning(pop)

// DCMTK
#include <dcmtk/dcmimgle/dcmimage.h>    // read helper (DicomImage)
#include <dcmtk/dcmdata/dctk.h>         // write helpers
#include <dcmtk/dcmdata/dcuid.h>

// OpenCV
#include <opencv2/imgcodecs.hpp>

// Your existing per-format loaders
#include "io_fastmri.hpp"
#include "io_ismrmrd.hpp"

// --- local dbg helper ---------------------------------------------------------
namespace {
template<class... Args>
void dbg_line(std::string* dbg, Args&&... a) {
    if (!dbg) return;
    std::ostringstream oss; (oss << ... << a);
    *dbg += oss.str(); *dbg += '\n';
}

static void h5_silence_once() {
    static bool done=false; if (done) return; done=true;
    H5::Exception::dontPrint();
    H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);
}

static std::string find_between(const std::string& s, const std::string& a, const std::string& b) {
    size_t i = s.find(a); if (i == std::string::npos) return "";
    i += a.size();
    size_t j = s.find(b, i); if (j == std::string::npos) return "";
    return s.substr(i, j - i);
}

static std::string read_xml_text_quick(H5::H5File& f) {
    try {
        if (!f.nameExists("/dataset/xml")) return {};
        H5::DataSet ds = f.openDataSet("/dataset/xml");
        H5::StrType mem(H5::PredType::C_S1, H5T_VARIABLE);
        char* cxml = nullptr; ds.read(&cxml, mem);
        std::string xml = cxml ? std::string(cxml) : std::string();
        if (cxml) free(cxml);
        return xml;
    } catch (...) { return {}; }
}
} // namespace

// --- public API ---------------------------------------------------------------
namespace io {

// -------------------- Probe --------------------
ProbeResult probe(const std::string& path, std::string* dbg) {
    ProbeResult pr;
    h5_silence_once();
    try {
        H5::H5File f(path, H5F_ACC_RDONLY);
        pr.has_xml = f.nameExists("/dataset/xml");
        pr.has_acq = f.nameExists("/dataset/acquisitions");
        pr.has_kspace = f.nameExists("kspace") || f.nameExists("/dataset/kspace");
        pr.has_embedded_img =
            f.nameExists("/dataset/reconstruction_rss") ||
            f.nameExists("/dataset/image") || f.nameExists("/dataset/images") ||
            f.nameExists("/dataset/image_0") || f.nameExists("/dataset/images_0");

        if (pr.has_xml || pr.has_acq) {
            const std::string xml = read_xml_text_quick(f);
            if (!xml.empty()) {
                const std::string enc = find_between(xml, "<encoding>", "</encoding>");
                const std::string tr  = find_between(enc, "<trajectory>", "</trajectory>");
                if (!tr.empty()) pr.trajectory = tr;
            }
            if (pr.trajectory == "cartesian") {
                pr.flavor = Flavor::ISMRMRD_Cartesian;
                pr.reason = "ISMRMRD xml/acquisitions; trajectory=cartesian";
            } else if (!pr.trajectory.empty()) {
                pr.flavor = Flavor::ISMRMRD_NonCartesian;
                pr.reason = "ISMRMRD xml/acquisitions; trajectory=" + pr.trajectory;
            } else {
                pr.flavor = Flavor::ISMRMRD_Unknown;
                pr.reason = "ISMRMRD bits present; trajectory unknown";
            }
            dbg_line(dbg, "[IO][Probe] HDF5 ISMRMRD: ", pr.reason);
            return pr;
        }

        if (pr.has_kspace) {
            pr.flavor = Flavor::FastMRI;
            pr.reason = "HDF5: top-level or /dataset/kspace present";
            dbg_line(dbg, "[IO][Probe] HDF5 fastMRI: ", pr.reason);
            return pr;
        }

        pr.flavor = Flavor::HDF5_Unknown;
        pr.reason = "HDF5 readable but no known markers";
        dbg_line(dbg, "[IO][Probe] HDF5 unknown layout");
        return pr;

    } catch (const H5::Exception&) {
        pr.flavor = Flavor::NotHDF5;
        pr.reason = "Not HDF5 (or unreadable as HDF5)";
        dbg_line(dbg, "[IO][Probe] Not an HDF5 file, trying DICOM...");

        // Quick DICOM sniff: try open with DicomImage (won’t throw)
        DicomImage di(path.c_str());
        if (di.getStatus() == EIS_Normal) {
            pr.flavor = Flavor::DICOM;
            pr.reason = "DicomImage opened OK";
            dbg_line(dbg, "[IO][Probe] Looks like DICOM");
        }
        return pr;
    }
}

// -------------------- Reading --------------------
bool load_hdf5_any(const std::string& path,
                   mri::KSpace& ks,
                   std::vector<float>* preRecon,
                   int* preH,
                   int* preW,
                   std::string* dbg)
{
    dbg_line(dbg, "[IO] load_hdf5_any: ", path);
    const auto pr = probe(path, dbg);

    bool ok = false;
    switch (pr.flavor) {
    case Flavor::FastMRI: {
        dbg_line(dbg, "[IO] Route: fastMRI loader");
        ok = mri::load_fastmri_kspace(path, ks, preRecon, preH, preW, dbg); // signature exists :contentReference[oaicite:0]{index=0}
        break;
    }
    case Flavor::ISMRMRD_Cartesian:
    case Flavor::ISMRMRD_Unknown:
    case Flavor::ISMRMRD_NonCartesian: {
        dbg_line(dbg, "[IO] Route: ISMRMRD loader");
        ok = mri::load_ismrmrd_any(path, ks, preRecon, preH, preW, dbg);    // signature exists :contentReference[oaicite:1]{index=1}
        break;
    }
    case Flavor::HDF5_Unknown:
    default: {
        dbg_line(dbg, "[IO][WARN] Unknown HDF5 layout -> fastMRI then ISMRMRD fallback");
        ok = mri::load_fastmri_kspace(path, ks, preRecon, preH, preW, dbg); // :contentReference[oaicite:2]{index=2}
        if (!ok) {
            dbg_line(dbg, "[IO] fastMRI failed -> try ISMRMRD");
            ok = mri::load_ismrmrd_any(path, ks, preRecon, preH, preW, dbg); // :contentReference[oaicite:3]{index=3}
        }
        break;
    }
    }
    dbg_line(dbg, "[IO] load_hdf5_any result: ", (ok?"OK":"FAIL"));
    return ok;
}

bool read_dicom_gray8(const std::string& path, cv::Mat& out8, std::string* why) {
    std::cerr << "[IO][DICOM] Opening: " << path << "\n";
    DicomImage di(path.c_str());
    if (di.getStatus() != EIS_Normal) {
        if (why) *why = "DicomImage status not normal";
        std::cerr << "[IO][DICOM][ERR] status=" << (int)di.getStatus() << "\n";
        return false;
    }
    if (di.isMonochrome()) di.setMinMaxWindow();

    const int w = static_cast<int>(di.getWidth());
    const int h = static_cast<int>(di.getHeight());
    std::cerr << "[IO][DICOM] dims=" << w << "x" << h
              << " frames=" << di.getFrameCount() << "\n";

    const int frame = 0;
    const void* pix = di.getOutputData(8 /*bits*/, frame);
    if (!pix) {
        if (why) *why = "getOutputData(8) returned null";
        std::cerr << "[IO][DICOM][ERR] getOutputData(8) null\n";
        return false;
    }

    out8 = cv::Mat(h, w, CV_8UC1);
    std::memcpy(out8.data, pix, static_cast<size_t>(w) * static_cast<size_t>(h));
    std::cerr << "[IO][DICOM] CV_8UC1 prepared.\n";
    return true;
}

bool io::write_png(const std::string& path_utf8, const cv::Mat& img8, std::string* why) {
    std::cerr << "[IO][PNG] write -> " << path_utf8 << "\n";
    if (img8.empty() || img8.type() != CV_8UC1) {
        if (why) *why = "expected non-empty CV_8UC1";
        std::cerr << "[IO][PNG][ERR] expected CV_8UC1\n";
        return false;
    }

    try {
        // Build a filesystem path that is UTF-8 aware on Windows
#if defined(_WIN32)
        std::filesystem::path p = std::filesystem::u8path(path_utf8);
#else
        std::filesystem::path p(path_utf8);
#endif

        // Ensure parent folder exists
        std::error_code ec_mk;
        std::filesystem::create_directories(p.parent_path(), ec_mk);
        if (ec_mk) {
            if (why) *why = "create_directories failed: " + ec_mk.message();
            std::cerr << "[IO][PNG][ERR] create_directories: " << ec_mk.message() << "\n";
            return false;
        }

        // Encode to PNG in memory
        std::vector<uchar> buf;
        std::vector<int> params = { cv::IMWRITE_PNG_COMPRESSION, 3 };
        const bool enc_ok = cv::imencode(".png", img8, buf, params);
        if (!enc_ok || buf.empty()) {
            if (why) *why = "cv::imencode(.png) failed";
            std::cerr << "[IO][PNG][ERR] imencode failed (buf.size=" << buf.size() << ")\n";
            return false;
        }
        std::cerr << "[IO][PNG] Encoded bytes: " << buf.size() << "\n";

        // Write bytes using filesystem-aware stream (handles wide path on MSVC)
        std::ofstream ofs(p, std::ios::binary);
        if (!ofs) {
            if (why) *why = "ofstream open failed";
            std::cerr << "[IO][PNG][ERR] ofstream open failed\n";
            return false;
        }
        ofs.write(reinterpret_cast<const char*>(buf.data()),
                  static_cast<std::streamsize>(buf.size()));
        ofs.close();

        // Verify on disk
        std::error_code ec_ex, ec_sz;
        const bool exists = std::filesystem::exists(p, ec_ex);
        const auto size   = std::filesystem::file_size(p, ec_sz);
        std::filesystem::path canon = std::filesystem::weakly_canonical(p, ec_ex);

        std::cerr << "[IO][PNG] Saved. exists=" << exists
                  << " size=" << (ec_sz ? -1LL : static_cast<long long>(size))
                  << " canon=\"" << canon.string() << "\"\n";
        if (!exists) {
            if (why) *why = "file not visible after write";
            return false;
        }
        return true;

    } catch (const std::exception& e) {
        if (why) *why = e.what();
        std::cerr << "[IO][PNG][EXC] " << e.what() << "\n";
        return false;
    }
}

bool write_dicom_sc_gray8(const std::string& path, const cv::Mat& img8, std::string* why) {
    std::cerr << "[IO][DICOM] write -> " << path << "\n";
    if (img8.empty() || img8.type() != CV_8UC1) {
        if (why) *why = "expected non-empty CV_8UC1";
        std::cerr << "[IO][DICOM][ERR] expected CV_8UC1\n";
        return false;
    }

    try {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());

        DcmFileFormat ff;
        DcmDataset* ds = ff.getDataset();

        char studyUID[128]  = {0};
        char seriesUID[128] = {0};
        char instUID[128]   = {0};
        dcmGenerateUniqueIdentifier(studyUID);
        dcmGenerateUniqueIdentifier(seriesUID);
        dcmGenerateUniqueIdentifier(instUID);

        ds->putAndInsertString(DCM_SOPClassUID,          UID_SecondaryCaptureImageStorage);
        ds->putAndInsertString(DCM_SOPInstanceUID,       instUID);
        ds->putAndInsertString(DCM_SpecificCharacterSet, "ISO_IR 192"); // UTF-8
        ds->putAndInsertString(DCM_PatientName,          "Anon^Patient");
        ds->putAndInsertString(DCM_PatientID,            "0000");
        ds->putAndInsertString(DCM_StudyInstanceUID,     studyUID);
        ds->putAndInsertString(DCM_SeriesInstanceUID,    seriesUID);
        ds->putAndInsertString(DCM_Modality,             "OT");

        const Uint16 rows = static_cast<Uint16>(img8.rows);
        const Uint16 cols = static_cast<Uint16>(img8.cols);
        ds->putAndInsertUint16(DCM_Rows,                      rows);
        ds->putAndInsertUint16(DCM_Columns,                   cols);
        ds->putAndInsertString(DCM_PhotometricInterpretation, "MONOCHROME2");
        ds->putAndInsertUint16(DCM_SamplesPerPixel,           1);
        ds->putAndInsertUint16(DCM_BitsAllocated,             8);
        ds->putAndInsertUint16(DCM_BitsStored,                8);
        ds->putAndInsertUint16(DCM_HighBit,                   7);
        ds->putAndInsertUint16(DCM_PixelRepresentation,       0);

        const Uint32 nbytes = static_cast<Uint32>(img8.total());
        const Uint8* src    = reinterpret_cast<const Uint8*>(img8.data);
        OFCondition st = ds->putAndInsertUint8Array(DCM_PixelData, src, nbytes);
        if (st.bad()) {
            if (why) *why = std::string("PixelData insert failed: ") + st.text();
            std::cerr << "[IO][DICOM][ERR] putAndInsertUint8Array: " << st.text() << "\n";
            return false;
        }

        // Write using explicit little endian
        OFCondition saveSt = ff.saveFile(path.c_str(), EXS_LittleEndianExplicit);
        if (saveSt.bad()) {
            if (why) *why = std::string("saveFile failed: ") + saveSt.text();
            std::cerr << "[IO][DICOM][ERR] saveFile: " << saveSt.text() << "\n";
            return false;
        }
        std::cerr << "[IO][DICOM] Saved.\n";
        return true;

    } catch (const std::exception& e) {
        if (why) *why = e.what();
        std::cerr << "[IO][DICOM][EXC] " << e.what() << "\n";
        return false;
    }
}

} // namespace io
