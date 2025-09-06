#include "app_controller.hpp"
#include "../model/io_fastmri.hpp"
#include "../view/mainwindow.hpp"
#include "../src/image_utils.hpp"
#include "../model/io_ismrmrd.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <sstream>

#include <QStringList>
#include<QDateTime>
// HDF5 (for the file probe)
#include <H5Cpp.h>
#include <H5Epublic.h>


#include <dcmtk/dcmimgle/dcmimage.h>      // DicomImage
#include <cstring>                         // std::memcpy


// ---------------- probe helpers (file-local) ----------------
namespace {

enum class DataFlavor {
    FastMRI,
    ISMRMRD_Cartesian,
    ISMRMRD_NonCartesian,
    ISMRMRD_Unknown,
    HDF5_Unknown,
    NotHDF5
};

static const char* flavor_str(DataFlavor f) {
    switch (f) {
    case DataFlavor::FastMRI:            return "FastMRI";
    case DataFlavor::ISMRMRD_Cartesian:  return "ISMRMRD_Cartesian";
    case DataFlavor::ISMRMRD_NonCartesian:return "ISMRMRD_NonCartesian";
    case DataFlavor::ISMRMRD_Unknown:    return "ISMRMRD_Unknown";
    case DataFlavor::HDF5_Unknown:       return "HDF5_Unknown";
    default:                              return "NotHDF5";
    }
}

struct ProbeResult {
    DataFlavor flavor = DataFlavor::HDF5_Unknown;
    std::string trajectory;        // "cartesian", "radial", ...
    bool has_xml = false;
    bool has_acq = false;
    bool has_kspace = false;
    bool has_embedded_img = false;
    std::string reason;
};

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

static ProbeResult probe_hdf5_flavor(const std::string& path) {
    ProbeResult pr;
    try {
        H5::H5File f(path, H5F_ACC_RDONLY);

        pr.has_xml = f.nameExists("/dataset/xml");
        pr.has_acq = f.nameExists("/dataset/acquisitions");
        pr.has_kspace = f.nameExists("kspace") || f.nameExists("/dataset/kspace");
        pr.has_embedded_img =
            f.nameExists("/dataset/reconstruction_rss") ||
            f.nameExists("/dataset/image") ||
            f.nameExists("/dataset/images") ||
            f.nameExists("/dataset/image_0") ||
            f.nameExists("/dataset/images_0");

        // Prefer ISMRMRD detection if xml/acq are present
        if (pr.has_xml || pr.has_acq) {
            const std::string xml = read_xml_text_quick(f);
            if (!xml.empty()) {
                std::string enc = find_between(xml, "<encoding>", "</encoding>");
                std::string tr  = find_between(enc, "<trajectory>", "</trajectory>");
                if (!tr.empty()) pr.trajectory = tr;
            }
            if (pr.trajectory == "cartesian") {
                pr.flavor = DataFlavor::ISMRMRD_Cartesian;
                pr.reason = "ISMRMRD xml/acquisitions present; trajectory=cartesian";
            } else if (!pr.trajectory.empty()) {
                pr.flavor = DataFlavor::ISMRMRD_NonCartesian;
                pr.reason = "ISMRMRD xml/acquisitions present; trajectory=" + pr.trajectory;
            } else {
                pr.flavor = DataFlavor::ISMRMRD_Unknown;
                pr.reason = "ISMRMRD bits present; trajectory unknown";
            }
            return pr;
        }

        // Otherwise, looks like fastMRI if kspace at top or /dataset/kspace
        if (pr.has_kspace) {
            pr.flavor = DataFlavor::FastMRI;
            pr.reason = "Top-level or /dataset/kspace dataset present";
            return pr;
        }

        pr.flavor = DataFlavor::HDF5_Unknown;
        pr.reason = "HDF5 readable but no known markers";
        return pr;

    } catch (const H5::Exception&) {
        pr.flavor = DataFlavor::NotHDF5;
        pr.reason = "Not an HDF5 or unreadable";
        return pr;
    }
}

// Simple DICOM → cv::Mat (8-bit MONO) using DCMTK's DicomImage
static bool read_dicom_gray8(const std::string& path, cv::Mat& out8, std::string* why_fail = nullptr)
{
    std::cerr << "[DBG][DICOM] Opening: " << path << "\n";
    DicomImage di(path.c_str());
    if (di.getStatus() != EIS_Normal) {
        if (why_fail) *why_fail = "DicomImage status not normal";
        std::cerr << "[ERR][DICOM] DicomImage status=" << (int)di.getStatus() << "\n";
        return false;
    }

    // Pick a reasonable window for display (min–max)
    if (di.isMonochrome()) {
        di.setMinMaxWindow();
    }

    const int w = static_cast<int>(di.getWidth());
    const int h = static_cast<int>(di.getHeight());
    std::cerr << "[DBG][DICOM] dims=" << w << "x" << h << " frames=" << di.getFrameCount() << "\n";

    // Get 8-bit output for frame #0
    const int frame = 0;
    const void* pix = di.getOutputData(8 /*bits*/, frame);
    if (!pix) {
        if (why_fail) *why_fail = "getOutputData(8) returned null";
        std::cerr << "[ERR][DICOM] getOutputData(8) returned null\n";
        return false;
    }

    out8 = cv::Mat(h, w, CV_8UC1);
    std::memcpy(out8.data, pix, static_cast<size_t>(w) * static_cast<size_t>(h));
    std::cerr << "[DBG][DICOM] Prepared CV_8UC1 buffer for view.\n";
    return true;
}

} // namespace

// ---------------- controller ----------------
AppController::AppController(MainWindow* view) : m_view(view) {}

void AppController::loadAndShow(const QString& h5_path_q)
{
    const std::string path = h5_path_q.toStdString();
    std::cerr << "[DBG][Controller] Loading file: " << path << "\n";

    // --- NEW: start collecting metadata lines
    QStringList meta;
    meta << ("Source: " + h5_path_q);
    meta << ("Loaded at: " + QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss"));

    constexpr bool kPreferPreRecon = false;

    std::vector<float> pre;
    int preH = 0, preW = 0;
    std::string dbg;
    bool ok = false;

    h5_silence_once();
    std::cerr << "[DBG][Controller] Probing file flavor...\n";
    const ProbeResult pr = probe_hdf5_flavor(path);

    meta << QString("Probe: %1 traj='%2'")
                .arg(flavor_str(pr.flavor))
                .arg(QString::fromStdString(pr.trajectory));
    meta << QString("Flags: xml=%1 acq=%2 kspace=%3 img=%4")
                .arg(pr.has_xml).arg(pr.has_acq).arg(pr.has_kspace).arg(pr.has_embedded_img);
    meta << QString("Reason: %1").arg(QString::fromStdString(pr.reason));

    // --- Route/load, but also populate 'meta' and push it to the view before returning

    switch (pr.flavor) {
    case DataFlavor::NotHDF5: {
        std::cerr << "[DBG][Controller] Not an HDF5 container. Trying DICOM reader...\n";
        cv::Mat dicom8;
        std::string why;

        if (read_dicom_gray8(path, dicom8, &why)) {
            meta << "Format: DICOM";
            meta << QString("Dims: %1x%2").arg(dicom8.cols).arg(dicom8.rows);
            m_view->setMetadata(meta);
            std::cerr << "[DBG][Controller] DICOM read OK. Sending to view.\n";
            m_view->setImage(dicom8);
            return;
        } else {
            meta << QString("DICOM read failed: %1").arg(QString::fromStdString(why));
            m_view->setMetadata(meta);
            std::cerr << "[ERR][Controller] DICOM read failed: " << why << "\n";
            m_view->setImage(imgutil::make_test_gradient(512, 512));
            return;
        }
    }

    case DataFlavor::FastMRI: {
        std::cerr << "[DBG][Controller] Using fastMRI loader.\n";
        ok = mri::load_fastmri_kspace(path, m_ks, &pre, &preH, &preW, &dbg);
        std::cerr << dbg;
        meta << "Format: HDF5 / fastMRI";
        break;
    }

    case DataFlavor::ISMRMRD_Cartesian:
    case DataFlavor::ISMRMRD_Unknown: {
        std::cerr << "[DBG][Controller] Using ISMRMRD loader (Cartesian or unknown traj).\n";
        ok = mri::load_ismrmrd_any(path, m_ks, &pre, &preH, &preW, &dbg);
        std::cerr << dbg;
        meta << "Format: HDF5 / ISMRMRD";
        break;
    }

    case DataFlavor::ISMRMRD_NonCartesian: {
        std::cerr << "[DBG][Controller] Non-Cartesian ISMRMRD -> k-space disabled; will try embedded image.\n";
        ok = mri::load_ismrmrd_any(path, m_ks, &pre, &preH, &preW, &dbg);
        std::cerr << dbg;
        meta << "Format: HDF5 / ISMRMRD (non-Cartesian)";
        break;
    }

    case DataFlavor::HDF5_Unknown:
    default: {
        std::cerr << "[DBG][Controller][WARN] Unrecognized HDF5 layout. Trying fastMRI then ISMRMRD as fallback.\n";
        ok = mri::load_fastmri_kspace(path, m_ks, &pre, &preH, &preW, &dbg);
        std::cerr << dbg;
        if (ok) meta << "Format: HDF5 (unknown) → fastMRI path";
        if (!ok) {
            dbg.clear();
            ok = mri::load_ismrmrd_any(path, m_ks, &pre, &preH, &preW, &dbg);
            std::cerr << dbg;
            if (ok) meta << "Format: HDF5 (unknown) → ISMRMRD path";
        }
        break;
    }
    }

    if (!ok) {
        meta << "No supported datasets found (fastMRI/ISMRMRD). Showing gradient.";
        m_view->setMetadata(meta);
        std::cerr << "[ERR][Controller] No supported datasets found (fastMRI/ISMRMRD).\n";
        m_view->setImage(imgutil::make_test_gradient(512, 512));
        return;
    }

    if (kPreferPreRecon && !pre.empty()) {
        meta << QString("PreRecon used: reconstruction_rss %1x%2").arg(preW).arg(preH);
        m_view->setMetadata(meta);
        std::cerr << "[DBG][Controller] Using pre-reconstructed image (" << preW << "x" << preH << ").\n";
        m_view->setImage(imgutil::to_8u(pre, preH, preW));
        return;
    }

    if ((m_ks.host.empty() || m_ks.coils <= 0 || m_ks.nx <= 0 || m_ks.ny <= 0) && !pre.empty()) {
        meta << QString("Embedded image used: %1x%2").arg(preW).arg(preH);
        m_view->setMetadata(meta);
        std::cerr << "[DBG][Controller] No usable k-space; displaying embedded image (" << preW << "x" << preH << ").\n";
        m_view->setImage(imgutil::to_8u(pre, preH, preW));
        return;
    }

    // GPU recon path
    meta << QString("K-space: C=%1 ny=%2 nx=%3").arg(m_ks.coils).arg(m_ks.ny).arg(m_ks.nx);
    std::string recon_dbg;
    if (!mri::ifft_rss_gpu(m_ks, m_image, &recon_dbg)) {
        meta << "Reconstruction: IFFT+RSS GPU FAILED";
        m_view->setMetadata(meta);
        std::cerr << "[ERR][Controller] Reconstruction failed.\n";

        if (!pre.empty()) {
            meta << QString("Fallback: preRecon %1x%2").arg(preW).arg(preH);
            m_view->setMetadata(meta);
            std::cerr << "[DBG][Controller] Falling back to pre-reconstructed image ("
                      << preW << "x" << preH << ").\n";
            m_view->setImage(imgutil::to_8u(pre, preH, preW));
            return;
        }
        m_view->setImage(imgutil::make_test_gradient(std::max(64, m_ks.ny), std::max(64, m_ks.nx)));
        return;
    }
    std::cerr << recon_dbg;
    meta << "Reconstruction: IFFT + RSS (GPU) OK";

    const int outH = std::min(m_ks.ny, m_ks.nx);
    const int outW = outH;
    meta << QString("Display crop: %1x%2 (square)").arg(outW).arg(outH);
    m_view->setMetadata(meta);

    m_view->setImage(imgutil::to_8u(m_image, outH, outW));
    std::cerr << "[DBG][Controller] Image sent to view.\n";
}
