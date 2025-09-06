#include "app_controller.hpp"
#include "../view/mainwindow.hpp"
#include "../src/image_utils.hpp"   // if you log/stretch elsewhere
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <sstream>
#include <omp.h>
#include <QElapsedTimer>
#include <QMetaObject>

#include <QDateTime>
#include <QTimer>
#include <QFileInfo>

#include <chrono>


// --------------------------------------------------------
// ctor: keep your existing signal wiring
// --------------------------------------------------------
AppController::AppController(MainWindow* view) : m_view(view)
{
#ifdef _OPENMP
    std::cerr << "[DBG] OpenMP ON _OPENMP=" << _OPENMP << "\n";
    std::cerr << "[DBG] OMP nthreads: " << omp_get_max_threads() << "\n";
#else
    std::cerr << "[DBG] OpenMP OFF\n";
#endif

    QObject::connect(m_view, &MainWindow::requestSavePNG,  m_view,
                     [this](const QString& p){
                         std::cerr << "[DBG][Controller] requestSavePNG received\n";
                         this->savePNG(p);
                     });

    QObject::connect(m_view, &MainWindow::requestSaveDICOM, m_view,
                     [this](const QString& p){
                         std::cerr << "[DBG][Controller] requestSaveDICOM received\n";
                         this->saveDICOM(p);
                     });
}

// ===============================
// Small helpers (clarity > perf)
// ===============================
cv::Mat AppController::to_u8(const cv::Mat& f32)
{
    std::cerr << "[DBG][Helper][to_u8] begin\n";
    if (f32.empty() || f32.type() != CV_32F) {
        std::cerr << "[ERR][Helper][to_u8] input empty or not CV_32F\n";
        return {};
    }
    double mn = 0.0, mx = 0.0;
    cv::minMaxLoc(f32, &mn, &mx);
    if (mx - mn < 1e-12) {
        std::cerr << "[DBG][Helper][to_u8] flat image; expanding range minimally\n";
        mx = mn + 1.0;
    }
    cv::Mat norm32, u8;
    f32.convertTo(norm32, CV_32F, 1.0 / (mx - mn), -mn / (mx - mn));
    norm32.convertTo(u8, CV_8U, 255.0);
    std::cerr << "[DBG][Helper][to_u8] end\n";
    return u8;
}

cv::Mat AppController::vecf32_to_u8(const std::vector<float>& v, int H, int W)
{
    std::cerr << "[DBG][Helper][vecf32_to_u8] H=" << H << " W=" << W
              << " v.size=" << v.size() << "\n";
    if (H <= 0 || W <= 0 || (int)v.size() != H * W) {
        std::cerr << "[ERR][Helper][vecf32_to_u8] size mismatch or bad dims\n";
        return {};
    }
    cv::Mat f32(H, W, CV_32F, const_cast<float*>(v.data())); // view
    cv::Mat u8 = to_u8(f32);
    return u8.clone(); // own memory
}

cv::Mat AppController::make_gradient(int H, int W)
{
    std::cerr << "[DBG][Helper][make_gradient] H=" << H << " W=" << W << "\n";
    cv::Mat g(H, W, CV_8UC1);
    for (int y = 0; y < H; ++y) {
        uint8_t* row = g.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x) row[x] = static_cast<uint8_t>((x + y) & 255);
    }
    return g;
}

// ===============================
// Load pipeline
// ===============================
void AppController::clearLoadState()
{
    std::cerr << "[DBG][Load] clearLoadState()\n";
    m_meta.clear();
    m_probe = io::ProbeResult{};
    m_ks = mri::KSpace{};
    m_pre.clear(); m_preH = m_preW = 0;
    m_display8.release();
}

void AppController::load(const QString& pathQ)
{
    m_sourcePathQ = pathQ;
    const std::string path = pathQ.toStdString();
    std::cerr << "[DBG][Load] load() path=" << path << "\n";

    // Tell the view a new image is coming (clears old state)
    if (m_view) {
        std::cerr << "[DBG][Controller] Notifying view: beginNewImageCycle()\n";
        m_view->beginNewImageCycle();
    }

    // Show busy cursor/status until we return from this function
    BusyScope busy(m_view, QString("Loading %1").arg(pathQ));


    QElapsedTimer tAll; tAll.start();
    qint64 msProbe = 0, msDicom = 0, msH5 = 0, msRecon = 0, msEmbed = 0;

    clearLoadState();
    m_meta << ("Source: " + pathQ);
    m_meta << ("When: " + QDateTime::currentDateTime().toString(Qt::ISODate));

    // 1) Probe
    {
        QElapsedTimer t; t.start();
        load_probe(path);
        msProbe = t.elapsed();
        std::cerr << "[DBG][Load][T] probe=" << msProbe << " ms\n";
    }

    // 2) Branch by flavor
    const auto f = m_probe.flavor;
    if (f == io::Flavor::DICOM || f == io::Flavor::NotHDF5) {
        QElapsedTimer t; t.start();
        if (!load_dicom(path)) {
            std::cerr << "[ERR][Load] DICOM load failed; preparing fallback\n";
            prepare_fallback();
        }
        msDicom = t.elapsed();
        m_meta << QString("Timing(ms): probe=%1 dicom=%2 total=%3")
                      .arg(msProbe).arg(msDicom).arg(tAll.elapsed());
        return;
    }

    // 3) HDF5 path
    {
        QElapsedTimer t; t.start();
        if (!load_hdf5(path)) {
            std::cerr << "[ERR][Load] HDF5 load failed; preparing fallback\n";
            prepare_fallback();
            m_meta << QString("Timing(ms): probe=%1 hdf5=%2 total=%3")
                          .arg(msProbe).arg(t.elapsed()).arg(tAll.elapsed());
            return;
        }
        msH5 = t.elapsed();
        std::cerr << "[DBG][Load][T] hdf5=" << msH5 << " ms\n";
    }

    // 4) Prefer GPU recon; fallbacks cascade
    {
        QElapsedTimer t; t.start();
        if (try_gpu_recon()) {
            msRecon = t.elapsed();
            std::cerr << "[DBG][Load][T] recon=" << msRecon << " ms\n";
            m_meta << QString("Timing(ms): probe=%1 hdf5=%2 recon=%3 total=%4")
                          .arg(msProbe).arg(msH5).arg(msRecon).arg(tAll.elapsed());
            return;
        }
    }
    {
        QElapsedTimer t; t.start();
        if (use_embedded_preview()) {
            msEmbed = t.elapsed();
            std::cerr << "[DBG][Load][T] embed=" << msEmbed << " ms\n";
            m_meta << QString("Timing(ms): probe=%1 hdf5=%2 embed=%3 total=%4")
                          .arg(msProbe).arg(msH5).arg(msEmbed).arg(tAll.elapsed());
            return;
        }
    }

    std::cerr << "[WRN][Load] No displayable image after HDF5; using gradient fallback\n";
    prepare_fallback();
    m_meta << QString("Timing(ms): probe=%1 hdf5=%2 total=%3")
                  .arg(msProbe).arg(msH5).arg(tAll.elapsed());
}


void AppController::load_probe(const std::string& path)
{
    std::cerr << "[DBG][Load][Probe] start\n";
    std::string dbg;
    m_probe = io::probe(path, &dbg);
    std::cerr << dbg; // already verbose

    auto flavorStr = [&]() {
        using F = io::Flavor;
        switch (m_probe.flavor) {
        case F::FastMRI:              return "FastMRI";
        case F::ISMRMRD_Cartesian:    return "ISMRMRD_Cartesian";
        case F::ISMRMRD_NonCartesian: return "ISMRMRD_NonCartesian";
        case F::ISMRMRD_Unknown:      return "ISMRMRD_Unknown";
        case F::HDF5_Unknown:         return "HDF5_Unknown";
        case F::DICOM:                return "DICOM";
        default:                      return "NotHDF5";
        }
    }();

    m_meta << QString("Probe: %1  traj='%2'")
                  .arg(flavorStr)
                  .arg(QString::fromStdString(m_probe.trajectory));
    std::cerr << "[DBG][Load][Probe] flavor=" << flavorStr
              << " traj='" << m_probe.trajectory << "'\n";
    if (m_view) {
        const QString file = QFileInfo(QString::fromStdString(path)).fileName();
        const QString title = QString("Glimpse MRI — %1 (%2)")
                                  .arg(file)
                                  .arg(flavorStr);
        m_view->setWindowTitle(title);
        std::cerr << "[DBG][Show] Title set: " << title.toStdString() << "\n";
    }


}

bool AppController::load_dicom(const std::string& path)
{
    std::cerr << "[DBG][Load][DICOM] Trying read_dicom_gray8...\n";
    cv::Mat dicom8; std::string why;
    if (!io::read_dicom_gray8(path, dicom8, &why)) {
        std::cerr << "[ERR][Load][DICOM] read failed: " << why << "\n";
        m_meta << QString("DICOM read failed: %1").arg(QString::fromStdString(why));
        return false;
    }
    m_meta << "Format: DICOM (8-bit display)";
    m_meta << QString("Dims: %1x%2").arg(dicom8.cols).arg(dicom8.rows);
    m_display8 = dicom8.clone();
    std::cerr << "[DBG][Load][DICOM] OK -> dims " << dicom8.cols << "x" << dicom8.rows << "\n";
    return true;
}

bool AppController::load_hdf5(const std::string& path)
{
    std::cerr << "[DBG][Load][HDF5] load_hdf5_any start...\n";
    std::string loadDbg;
    bool ok = io::load_hdf5_any(path, m_ks, &m_pre, &m_preH, &m_preW, &loadDbg);
    std::cerr << loadDbg;

    if (!ok) {
        m_meta << "HDF5 load failed (no supported datasets).";
        std::cerr << "[ERR][Load][HDF5] load_hdf5_any failed\n";
        return false;
    }

    if (!m_ks.host.empty()) {
        m_meta << QString("K-space: coils=%1 ny=%2 nx=%3 buf=%4")
        .arg(m_ks.coils).arg(m_ks.ny).arg(m_ks.nx)
            .arg((qulonglong)m_ks.host.size());
        std::cerr << "[DBG][Load][HDF5] K-space: coils=" << m_ks.coils
                  << " ny=" << m_ks.ny << " nx=" << m_ks.nx
                  << " host.size=" << m_ks.host.size() << "\n";
    } else {
        std::cerr << "[DBG][Load][HDF5] No k-space present\n";
    }

    if (!m_pre.empty()) {
        m_meta << QString("Embedded image: %1x%2").arg(m_preW).arg(m_preH);
        std::cerr << "[DBG][Load][HDF5] Embedded preview: " << m_preW << "x" << m_preH << "\n";
    }
    return true;
}

bool AppController::try_gpu_recon()
{
    if (m_ks.host.empty()) {
        std::cerr << "[DBG][Load][GPU] No k-space; skipping GPU recon\n";
        return false;
    }

    std::cerr << "[DBG][Load][GPU] Running GPU IFFT+RSS...\n";
    std::vector<float> rssf;
    int outH = 0, outW = 0;
    std::string reconDbg;
    bool gpu_ok = false;

    try {
        gpu_ok = mri::ifft_rss_gpu(m_ks, rssf, outH, outW, &reconDbg);
    } catch (const std::exception& e) {
        std::cerr << "[EXC][Load][GPU] ifft_rss_gpu threw: " << e.what() << "\n";
        gpu_ok = false;
    } catch (...) {
        std::cerr << "[EXC][Load][GPU] ifft_rss_gpu threw unknown exception\n";
        gpu_ok = false;
    }
    std::cerr << reconDbg;

    if (!gpu_ok) {
        std::cerr << "[ERR][Load][GPU] GPU recon failed\n";
        return false;
    }

    const int N = outH * outW;
    if ((int)rssf.size() != N || outH <= 0 || outW <= 0) {
        std::cerr << "[ERR][Load][GPU] Bad dims from GPU: H=" << outH
                  << " W=" << outW << " rss.size=" << rssf.size()
                  << " expected=" << N << "\n";
        return false;
    }

    cv::Mat u8 = vecf32_to_u8(rssf, outH, outW);
    if (u8.empty()) {
        std::cerr << "[ERR][Load][GPU] Conversion to 8-bit failed\n";
        return false;
    }

    m_meta << QString("Display: GPU IFFT + RSS (%1x%2)").arg(outW).arg(outH);
    if (outH != m_ks.ny || outW != m_ks.nx) {
        m_meta << QString("Note: GPU returned %1x%2 (input k-space %3x%4)")
        .arg(outW).arg(outH).arg(m_ks.nx).arg(m_ks.ny);
    }
    m_display8 = u8.clone();
    std::cerr << "[DBG][Load][GPU] Recon OK; dims " << outW << "x" << outH << "\n";
    return true;
}

bool AppController::use_embedded_preview()
{
    if (m_pre.empty() || m_preH <= 0 || m_preW <= 0) {
        std::cerr << "[DBG][Load][Embed] No embedded preview available\n";
        return false;
    }
    std::cerr << "[DBG][Load][Embed] Using embedded preview fallback\n";
    cv::Mat f32(m_preH, m_preW, CV_32F, m_pre.data());
    cv::Mat u8 = to_u8(f32);
    if (u8.empty()) {
        std::cerr << "[ERR][Load][Embed] to_u8 returned empty image\n";
        return false;
    }
    m_meta << "Display: Embedded pre-recon (fallback)";
    m_display8 = u8.clone();
    return true;
}

void AppController::prepare_fallback()
{
    std::cerr << "[DBG][Load] Preparing gradient fallback\n";
    m_meta << "Display: fallback gradient";
    m_display8 = make_gradient(512, 512);
}

// ===============================
// Show pipeline
// ===============================
void AppController::show()
{
    std::cerr << "[DBG][Show] show() -> defer via QTimer::singleShot(0ms)\n";
    using namespace std::chrono_literals; // enables 0ms literal
    QTimer::singleShot(0ms, m_view, [this]() {
        std::cerr << "[DBG][Show] (deferred) entering doShowNow()\n";
        this->doShowNow();
    });
}


void AppController::doShowNow()
{
    if (m_display8.empty() || m_display8.type() != CV_8UC1) {
        std::cerr << "[WRN][Show] m_display8 empty; forcing gradient fallback in doShowNow()\n";
        QStringList meta = m_meta;
        meta << "Display: forced gradient (no valid 8-bit image)";
        show_gradient_with_meta(meta);
        return;
    }

    // (Optional) set title here so you can see it appear with the image
    // m_view->setWindowTitle(...); // same code we discussed earlier

    show_metadata_and_image(m_meta, m_display8); // this calls into the View
    std::cerr << "[DBG][Show] doShowNow() complete\n";
}


void AppController::show_metadata_and_image(const QStringList& meta, const cv::Mat& u8)
{
    std::cerr << "[DBG][Show] show_metadata_and_image() dims=" << u8.cols << "x" << u8.rows << "\n";
    m_view->setMetadata(meta);
    m_view->setImage(u8);
    m_lastImg8 = u8.clone();
    std::cerr << "[DBG][Show] Image pushed to view\n";
}


void AppController::show_gradient_with_meta(const QStringList& meta)
{
    std::cerr << "[DBG][Show] show_gradient_with_meta()\n";
    cv::Mat grad = make_gradient(512, 512);
    m_view->setMetadata(meta);
    m_view->setImage(grad);
    m_lastImg8 = grad.clone();
}

// ===============================
// Save operations (unchanged)
// ===============================
void AppController::savePNG(const QString& outPath)
{
    std::cerr << "[DBG][Controller] savePNG -> " << outPath.toStdString() << "\n";
    if (m_lastImg8.empty() || m_lastImg8.type() != CV_8UC1) {
        std::cerr << "[ERR][Controller] savePNG: no 8-bit image available to save.\n";
        return;
    }
    const std::string path_utf8 = outPath.toUtf8().constData(); // UTF-8 for OpenCV
    std::string why;
    if (!io::write_png(path_utf8, m_lastImg8, &why)) {
        std::cerr << "[ERR][Controller] PNG save failed: " << why << "\n";
    } else {
        std::cerr << "[DBG][Controller] PNG saved OK.\n";
    }
}

void AppController::saveDICOM(const QString& outPath)
{
    std::cerr << "[DBG][Controller] saveDICOM -> " << outPath.toStdString() << "\n";
    if (m_lastImg8.empty() || m_lastImg8.type() != CV_8UC1) {
        std::cerr << "[ERR][Controller] saveDICOM: no 8-bit image available to save.\n";
        return;
    }
    const std::string path_local8 = outPath.toLocal8Bit().constData(); // local 8-bit for DCMTK
    std::string why;
    if (!io::write_dicom_sc_gray8(path_local8, m_lastImg8, &why)) {
        std::cerr << "[ERR][Controller] DICOM save failed: " << why << "\n";
    } else {
        std::cerr << "[DBG][Controller] DICOM saved OK.\n";
    }
}
