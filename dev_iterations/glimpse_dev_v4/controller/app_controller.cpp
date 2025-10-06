// controller/app_controller.cpp
#include "app_controller.hpp"

// From the DLL include folder (added to INCLUDEPATH in your .pro)
#include "engine_api.h"   // NEW public API (engine_init/engine_reconstruct_all/engine_free)

#include <mutex>              // std::once_flag
#include <vector>             // std::vector
#include <cstring>            // std::memcpy

#include "../view/mainwindow.hpp"
#include "../model/io.hpp"
#include "../src/image_utils.hpp"    // imgutil::to_u8_slice / make_test_gradient if needed

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QMetaObject>
#include <QTimer>
#include <QShortcut>
#include <QLibrary>
#include <QDebug>   // <-- added

// ============================================================
// BusyScope out-of-line (needs complete MainWindow type)
// ============================================================
AppController::BusyScope::BusyScope(MainWindow* v, const QString& m) : v_(v) {
    if (v_) v_->beginBusy(m);
}
AppController::BusyScope::~BusyScope() {
    if (v_) v_->endBusy();
}

AppController::AppController(MainWindow* view) : m_view(view)
{
#ifdef _OPENMP
    qDebug() << "[DBG] OpenMP ON _OPENMP=" << _OPENMP
             << "  (max threads=" << omp_get_max_threads() << ")";
#else
    qDebug() << "[DBG] OpenMP OFF";
#endif

    if (!m_view) {
        qWarning() << "[ERR][Controller] MainWindow pointer is null; skipping UI wiring.";
        return;
    }

    // ---- Wire view -> controller for file saves ----
    QObject::connect(m_view, &MainWindow::requestSavePNG,  m_view,
                     [this](const QString& p){ this->savePNG(p); });
    QObject::connect(m_view, &MainWindow::requestSaveDICOM, m_view,
                     [this](const QString& p){ this->saveDICOM(p); });

    // ---- Slice change (View -> Controller) ----
    QObject::connect(m_view, &MainWindow::sliceChanged, m_view,
                     [this](int idx) { this->onSliceChanged(idx); });

    // Shortcuts
    auto step = [this](int d){
        if (m_slices8.empty()) return;
        const int n = (int)m_slices8.size();
        showSlice(std::clamp(m_currentSlice + d, 0, n-1));
    };
    auto jump = [this](int t){
        if (m_slices8.empty()) return;
        const int n = (int)m_slices8.size();
        showSlice(std::clamp(t, 0, n-1));
    };
    auto bindStep = [&](const QKeySequence& ks, int d){
        auto* sc = new QShortcut(ks, m_view);
        QObject::connect(sc, &QShortcut::activated, m_view, [=]{ step(d); });
    };
    auto bindJump = [&](const QKeySequence& ks, int t){
        auto* sc = new QShortcut(ks, m_view);
        QObject::connect(sc, &QShortcut::activated, m_view, [=]{ jump(t); });
    };
    bindStep(QKeySequence(Qt::Key_Up), -1);
    bindStep(QKeySequence(Qt::Key_Left), -1);
    bindStep(QKeySequence(Qt::Key_Down), +1);
    bindStep(QKeySequence(Qt::Key_Right), +1);
    bindStep(QKeySequence(Qt::Key_PageUp), -5);
    bindStep(QKeySequence(Qt::Key_PageDown), +5);
    bindJump(QKeySequence(Qt::Key_Home), 0);
    bindJump(QKeySequence(Qt::Key_End), 1'000'000);
    bindStep(QKeySequence("["), -1);
    bindStep(QKeySequence("]"), +1);
}

// ===============================
// Helpers
// ===============================
cv::Mat AppController::to_u8(const cv::Mat& f32)
{
    if (f32.empty() || f32.type() != CV_32F) return {};
    double mn=0.0, mx=0.0; cv::minMaxLoc(f32, &mn, &mx);
    if (mx - mn < 1e-12) mx = mn + 1.0;
    cv::Mat norm32, u8;
    f32.convertTo(norm32, CV_32F, 1.0/(mx-mn), -mn/(mx-mn));
    norm32.convertTo(u8, CV_8U, 255.0);
    return u8;
}

cv::Mat AppController::vecf32_to_u8(const std::vector<float>& v, int H, int W)
{
    if (H <= 0 || W <= 0 || (int)v.size() != H*W) return {};
    cv::Mat f32(H, W, CV_32F, const_cast<float*>(v.data()));
    return to_u8(f32).clone();
}

cv::Mat AppController::make_gradient(int H, int W)
{
    cv::Mat g(H, W, CV_8UC1);
    for (int y=0; y<H; ++y) {
        uint8_t* row = g.ptr<uint8_t>(y);
        for (int x=0; x<W; ++x) row[x] = uint8_t((x + y) & 255);
    }
    return g;
}

// ===============================
// Load pipeline
// ===============================
void AppController::clearLoadState()
{
    m_meta.clear();
    m_probe = io::ProbeResult{};
    m_pre.clear(); m_preH = m_preW = 0;
    m_display8.release();
    m_lastImg8.release();
    m_slices8.clear();
    m_currentSlice = 0;
}

void AppController::load(const QString& pathQ)
{
    m_sourcePathQ = pathQ;
    const std::string path = pathQ.toStdString();

    if (m_view) m_view->beginNewImageCycle();
    BusyScope busy(m_view, QString("Loading %1").arg(pathQ));

    QElapsedTimer tAll; tAll.start();
    qint64 msProbe=0, msDicom=0, msRecon=0;

    clearLoadState();
    m_meta << ("Source: " + pathQ);
    m_meta << ("When: " + QDateTime::currentDateTime().toString(Qt::ISODate));

    // 1) Probe (extension/magic only)
    std::string dbg;
    { QElapsedTimer t; t.start(); m_probe = io::probe(path, &dbg); msProbe = t.elapsed(); }
    qDebug().noquote() << "[DBG][probe]\n" << QString::fromStdString(dbg);

    // 2) DICOM path
    if (m_probe.flavor == io::Flavor::DICOM) {
        QElapsedTimer t; t.start();
        if (!load_dicom(path)) prepare_fallback();
        msDicom = t.elapsed();
        m_meta << QString("Timing(ms): probe=%1 dicom=%2 total=%3")
                      .arg(msProbe).arg(msDicom).arg(tAll.elapsed());
        show();
        return;
    }

    // 3) HDF5 → DLL (single path: let the engine figure out FastMRI/ISMRMRD)
    {
        QElapsedTimer t; t.start();
        const bool ok = reconstructAllSlicesFromDll(pathQ, /*fftshift=*/true);
        msRecon = t.elapsed();

        if (ok) {
            m_meta << "Display: DLL GPU recon (full stack)";
            m_meta << QString("Timing(ms): probe=%1 dll_recon=%2 total=%3")
                          .arg(msProbe).arg(msRecon).arg(tAll.elapsed());
            show();
            return;
        }
        qWarning() << "[DBG][DLL] recon failed; falling back to gradient.";
    }

    // 4) Last resort
    prepare_fallback();
    m_meta << QString("Timing(ms): probe=%1 total=%2").arg(msProbe).arg(tAll.elapsed());
    show();
}

void AppController::load_probe(const std::string& path)
{
    std::string dbg;
    m_probe = io::probe(path, &dbg); // io::probe already logs
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
}

bool AppController::load_dicom(const std::string& path)
{
    qDebug() << "[DBG][DICOM] load_dicom path =" << QString::fromStdString(path);

    std::vector<cv::Mat> frames;
    std::string why;
    if (!io::read_dicom_frames_gray8(path, frames, &why) || frames.empty()) {
        m_meta << QString("DICOM read failed: %1").arg(QString::fromStdString(why));
        qWarning() << "[DBG][DICOM] read_dicom_frames_gray8 failed:" << QString::fromStdString(why);
        return false;
    }

    // --- Basic DICOM metadata (Manufacturer, Model, SW, B0, TR/TE/TI, etc.) ---
    {
        io::DicomMeta dm;
        std::string whyMeta;
        if (io::read_dicom_basic_meta(path, dm, &whyMeta)) {
            if (!dm.manufacturer.empty())      m_meta << "Manufacturer: " + QString::fromStdString(dm.manufacturer);
            if (!dm.modelName.empty())         m_meta << "Model: " + QString::fromStdString(dm.modelName);
            if (!dm.institutionName.empty())   m_meta << "Institution: " + QString::fromStdString(dm.institutionName);
            if (!dm.B0T.empty())               m_meta << "Field (T): " + QString::fromStdString(dm.B0T);
            if (!dm.softwareVersions.empty())  m_meta << "Software: " + QString::fromStdString(dm.softwareVersions);
            if (!dm.seriesDescription.empty()) m_meta << "Series: " + QString::fromStdString(dm.seriesDescription);

            // Optional timing params (shown only if present)
            QStringList seq;
            if (!dm.tr_ms.empty()) seq << ("TR (ms): " + QString::fromStdString(dm.tr_ms));
            if (!dm.te_ms.empty()) seq << ("TE (ms): " + QString::fromStdString(dm.te_ms));
            if (!dm.ti_ms.empty()) seq << ("TI (ms): " + QString::fromStdString(dm.ti_ms));
            if (!seq.isEmpty()) m_meta << seq;

            if (!dm.studyDate.empty() || !dm.studyTime.empty()) {
                m_meta << QString("Study: %1 %2")
                .arg(QString::fromStdString(dm.studyDate))
                    .arg(QString::fromStdString(dm.studyTime));
            }
            if (!dm.patientName.empty() || !dm.patientID.empty()) {
                m_meta << QString("Patient: %1  ID=%2")
                .arg(QString::fromStdString(dm.patientName))
                    .arg(QString::fromStdString(dm.patientID));
            }

            qDebug() << "[DBG][DICOM][META] manufacturer="
                     << QString::fromStdString(dm.manufacturer)
                     << " model=" << QString::fromStdString(dm.modelName)
                     << " B0T="   << QString::fromStdString(dm.B0T)
                     << " TR/TE/TI(ms)="
                     << QString::fromStdString(dm.tr_ms) << "/"
                     << QString::fromStdString(dm.te_ms) << "/"
                     << QString::fromStdString(dm.ti_ms);
        } else {
            qWarning() << "[DBG][DICOM][META] read failed:"
                       << QString::fromStdString(whyMeta);
        }
    }
    // --------------------------------------------------------------------------

    if (frames.size() >= 2) {
        m_meta << QString("Format: DICOM (multi-slice)  S=%1").arg((qulonglong)frames.size());
        m_meta << QString("Dims: %1x%2").arg(frames[0].cols).arg(frames[0].rows);
        qDebug() << "[DBG][DICOM] multi-slice frames=" << (qulonglong)frames.size()
                 << " dims=" << frames[0].cols << "x" << frames[0].rows;
        showSlices(frames);
        return true;
    }

    m_meta << "Format: DICOM (single frame)";
    m_meta << QString("Dims: %1x%2").arg(frames[0].cols).arg(frames[0].rows);
    qDebug() << "[DBG][DICOM] single-frame dims=" << frames[0].cols << "x" << frames[0].rows;
    m_display8 = frames[0].clone();
    return true;
}

bool AppController::load_hdf5(const std::string& path)
{
    Q_UNUSED(path);
    qDebug() << "[DBG][HDF5] Qt-side HDF5 loader disabled; delegating to DLL.";
    // We no longer parse HDF5 on the Qt side. The DLL will do IO+recon.
    // Clear any stale preview info.
    m_pre.clear();
    m_preH = 0;
    m_preW = 0;

    // Optional: annotate UI
    m_meta << "HDF5: delegated to MRI engine DLL";
    return true; // allow the load() pipeline to proceed to DLL recon
}

#ifdef MRI_INPROC_FALLBACK
bool AppController::recon_inproc()
{
    std::vector<float> rssf; int h=0, w=0; std::string reconDbg;
    bool ok = false;
    try { ok = mri::ifft_rss_gpu(m_ks, rssf, h, w, &reconDbg); }
    catch (...) { ok = false; }
    if (!ok || h<=0 || w<=0 || rssf.size() != size_t(h)*size_t(w)) return false;
    m_display8 = vecf32_to_u8(rssf, h, w);
    m_meta << QString("Display: GPU IFFT + RSS (in-process) %1x%2").arg(w).arg(h);
    return !m_display8.empty();
}
#endif

bool AppController::use_embedded_preview()
{
    if (m_pre.empty() || m_preH <= 0 || m_preW <= 0) return false;
    const size_t HW = size_t(m_preH)*size_t(m_preW);
    const size_t N  = m_pre.size();
    const size_t S  = (HW && (N % HW == 0)) ? (N / HW) : 0;

    if (S >= 2) {
        std::vector<cv::Mat> slices;
        slices.reserve(S);
        for (size_t s=0; s<S; ++s) {
            float* base = m_pre.data() + s*HW;
            cv::Mat f32(m_preH, m_preW, CV_32F, base);
            cv::Mat u8 = to_u8(f32);
            if (!u8.empty()) slices.push_back(u8.clone());
        }
        if (!slices.empty()) {
            m_meta << QString("Display: Embedded stack (S=%1) %2x%3")
            .arg((qulonglong)slices.size()).arg(m_preW).arg(m_preH);
            m_slices8 = std::move(slices);
            m_display8.release();
            return true;
        }
    }

    cv::Mat f32(m_preH, m_preW, CV_32F, m_pre.data());
    cv::Mat u8 = to_u8(f32);
    if (u8.empty()) return false;
    m_meta << "Display: Embedded pre-recon (single)";
    m_display8 = u8.clone();
    return true;
}

void AppController::prepare_fallback()
{
    m_meta << "Display: fallback gradient";
    m_display8 = make_gradient(512, 512);
}

// ===============================
// Show pipeline
// ===============================
void AppController::show()
{
    using namespace std::chrono_literals;
    QTimer::singleShot(0ms, m_view, [this](){ this->doShowNow(); });
}

void AppController::doShowNow()
{
    if (!m_slices8.empty()) {
        if (m_view) m_view->enableSliceSlider((int)m_slices8.size());
        showSlice(std::clamp(m_currentSlice, 0, (int)m_slices8.size()-1));
        return;
    }
    if (m_display8.empty() || m_display8.type() != CV_8UC1) {
        QStringList meta = m_meta; meta << "Display: forced gradient (no valid 8-bit image)";
        show_gradient_with_meta(meta);
        return;
    }
    if (m_view) m_view->enableSliceSlider(0);
    show_metadata_and_image(m_meta, m_display8);
}

void AppController::show_metadata_and_image(const QStringList& meta, const cv::Mat& u8)
{
    m_view->setMetadata(meta);
    m_view->setImage(u8);
    m_lastImg8 = u8.clone();
}

void AppController::show_gradient_with_meta(const QStringList& meta)
{
    cv::Mat grad = make_gradient(512, 512);
    m_view->setMetadata(meta);
    m_view->setImage(grad);
    m_lastImg8 = grad.clone();
}

// ===============================
// Multi-slice helpers
// ===============================
void AppController::adoptReconStackF32(const std::vector<float>& stack, int S, int H, int W)
{
    m_slices8.clear();
    if (S <= 0 || H <= 0 || W <= 0 || stack.size() < (size_t)S*H*W) {
        if (m_view) m_view->enableSliceSlider(S);
        return;
    }
    const size_t HW = (size_t)H*W;
    m_slices8.reserve(S);
    for (int s=0; s<S; ++s) {
        const float* src = stack.data() + (size_t)s*HW;
        // Use shared helper
        m_slices8.emplace_back(imgutil::to_u8_slice(src, H, W));
    }
    if (m_view) { m_view->enableSliceSlider(S); m_view->setSliceIndex(0); }
    m_currentSlice = 0;
    if (!m_slices8.empty()) showSlice(0);
}

void AppController::showSlices(const std::vector<cv::Mat>& slices)
{
    m_slices8.clear();
    m_currentSlice = 0;
    for (const auto& s : slices) {
        if (s.empty()) continue;
        if (s.type() == CV_8UC1) m_slices8.push_back(s.clone());
        else if (s.type() == CV_32F) m_slices8.push_back(to_u8(s));
    }
    if (m_view) m_view->enableSliceSlider((int)m_slices8.size());
    showSlice(m_currentSlice);
}

void AppController::showSlice(int idx)
{
    if (m_slices8.empty()) { doShowNow(); return; }
    if (idx < 0 || idx >= (int)m_slices8.size()) return;
    m_currentSlice = idx;
    if (m_view) m_view->setSliceIndex(idx);
    QStringList meta = m_meta; meta << QString("Slice %1 / %2").arg(idx+1).arg(m_slices8.size());
    show_metadata_and_image(meta, m_slices8[idx]);
}

// ===============================
// Save operations
// ===============================
void AppController::savePNG(const QString& outPath)
{
    if (m_lastImg8.empty() || m_lastImg8.type() != CV_8UC1) return;
    const std::string path_utf8 = outPath.toUtf8().constData();
    std::string why;
    io::write_png(path_utf8, m_lastImg8, &why);
}

void AppController::saveDICOM(const QString& outPath)
{
    if (m_lastImg8.empty() || m_lastImg8.type() != CV_8UC1) return;
    const std::string path_local8 = outPath.toLocal8Bit().constData();
    std::string why;
    io::write_dicom_sc_gray8(path_local8, m_lastImg8, &why);
}

struct AppController::MriEngineDll {
    using PFN_init = int (*)(int);
    using PFN_ifft = int (*)(const float*, int, int, int, float*, int*, int*, char*, int);

    QLibrary lib;
    PFN_init p_init = nullptr;
    PFN_ifft p_ifft = nullptr;
    bool loaded = false;

    bool load(const QString& explicitPath);  // your existing implementation
};

AppController::~AppController() = default;

void AppController::onSliceChanged(int idx)
{
    if (m_slices8.empty()) {
        qWarning() << "[CTRL][WRN] onSliceChanged: no slices; ignoring";
        return;
    }
    if (idx < 0 || idx >= (int)m_slices8.size()) {
        qWarning() << "[CTRL][WRN] onSliceChanged: index" << idx
                   << "out of range [0," << (int)m_slices8.size()-1 << "]";
        return;
    }
    m_currentSlice = idx;
    if (m_view) {
        m_view->setSliceIndex(idx);   // keep slider in sync (no loop if view blocks signal)
        m_view->setImage(m_slices8[idx]);
    }
}

// DLL, add lib on .pro or CMake
bool AppController::reconstructAllSlicesFromDll(const QString& pathQ, bool fftshift)
{
    qDebug() << "[DBG][DLL] reconstructAllSlicesFromDll path=" << pathQ << " fftshift=" << fftshift;

    // 1) Init the engine exactly once (clarity over perf)
    static std::once_flag s_once;
    static int s_init_ok = 0;
    std::call_once(s_once, [&](){

        const char* ver = engine_version();
        qDebug() << "[DBG][DLL] engine_version ->" << (ver ? ver : "(null)");
        // device_id:
        //   0  -> Auto (try CUDA then CPU)
        //  -1  -> Force CPU (same as MRI_FORCE_CPU set)
        s_init_ok = engine_init(0);

        qDebug() << "[DBG][DLL] engine_init ->" << s_init_ok;
    });
    if (!s_init_ok) {
        qWarning() << "[DBG][DLL] engine_init failed; skipping DLL path";
        return false;
    }

    // 2) Call DLL
    int S = 0, H = 0, W = 0;
    float* stack = nullptr;
    char dbg[4096] = {0};

    const QByteArray path8 = pathQ.toUtf8();
    const int ok = engine_reconstruct_all(
        path8.constData(),
        &S, &H, &W,
        &stack,
        fftshift ? 1 : 0,
        dbg, int(sizeof(dbg)));

    // 3) Log DLL-side debug (the DLL also logs to stderr; we capture here too)
    if (dbg[0] != '\0') {
        qDebug().noquote() << "[DBG][DLL] " << dbg;
    }

    if (!ok || !stack || S <= 0 || H <= 0 || W <= 0) {
        qWarning() << "[DBG][DLL] reconstruct_all failed or returned invalid dims"
                   << " ok=" << ok << " stack=" << (void*)stack
                   << " S=" << S << " H=" << H << " W=" << W;
        if (stack) engine_free(stack);
        return false;
    }

    // 4) Copy to host vector (clarity > perf) and free engine buffer
    const size_t count = size_t(S) * size_t(H) * size_t(W);
    std::vector<float> host(count);
    std::memcpy(host.data(), stack, count * sizeof(float));
    engine_free(stack);

    // 5) Hand off to your existing adopt method and add some meta
    adoptReconStackF32(host, S, H, W);
    m_meta << QString("DLL: Slices=%1, Size=%2x%3").arg(S).arg(W).arg(H);

    // 6) Optional: read ISMRMRD/HDF5 metadata for UI (Manufacturer, Field, TR/TE/TI)
    {
        io::DicomMeta dm;
        std::string whyMeta;
        const std::string path = pathQ.toStdString();
        if (io::read_hdf5_ismrmrd_meta(path, dm, &whyMeta)) {
            if (!dm.manufacturer.empty())     m_meta << "Manufacturer: " + QString::fromStdString(dm.manufacturer);
            if (!dm.modelName.empty())        m_meta << "Model: " + QString::fromStdString(dm.modelName);
            if (!dm.institutionName.empty())  m_meta << "Institution: " + QString::fromStdString(dm.institutionName);
            if (!dm.B0T.empty())              m_meta << "Field (T): " + QString::fromStdString(dm.B0T);
            if (!dm.tr_ms.empty() || !dm.te_ms.empty() || !dm.ti_ms.empty()) {
                const QString tr = QString::fromStdString(dm.tr_ms);
                const QString te = QString::fromStdString(dm.te_ms);
                const QString ti = QString::fromStdString(dm.ti_ms);
                m_meta << QString("TR/TE/TI (ms): %1 / %2 / %3")
                              .arg(tr.isEmpty() ? "-" : tr)
                              .arg(te.isEmpty() ? "-" : te)
                              .arg(ti.isEmpty() ? "-" : ti);
            }
            qDebug() << "[DBG][H5][META] TR/TE/TI(ms)="
                     << QString::fromStdString(dm.tr_ms) << "/"
                     << QString::fromStdString(dm.te_ms) << "/"
                     << QString::fromStdString(dm.ti_ms);
        } else if (!whyMeta.empty()) {
            qWarning() << "[DBG][H5][META] read failed:" << QString::fromStdString(whyMeta);
        }
    }

    qDebug() << "[DBG][DLL] reconstructAllSlicesFromDll OK: S=" << S << " H=" << H << " W=" << W;
    return true;
}
