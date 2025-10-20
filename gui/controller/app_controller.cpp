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


// app_controller.hpp
#include "../model/dicom_dll.hpp"

std::unique_ptr<DicomDll> m_dicom;  // NEW


// ============================================================
// BusyScope out-of-line (needs complete MainWindow type)
// ============================================================
AppController::AppController(MainWindow* view) : m_view(view)
{

    qDebug() << "[DBG] Qt app starting.";

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
                     [this](const QString& p){ qDebug() << "[CTRL] savePNG ->" << p; this->savePNG(p); });
    QObject::connect(m_view, &MainWindow::requestSaveDICOM, m_view,
                     [this](const QString& p){ qDebug() << "[CTRL] saveDICOM ->" << p; this->saveDICOM(p); });

    // ---- Slice change (View -> Controller) ----
    QObject::connect(m_view, &MainWindow::sliceChanged, m_view,
                     [this](int idx) { qDebug() << "[CTRL] sliceChanged ->" << idx; this->onSliceChanged(idx); });

    // ---- Drag-and-drop (View -> Controller) ----
    QObject::connect(m_view, &MainWindow::fileDropped, m_view,
                     [this](const QString& p){
                         qDebug() << "[DnD][CTRL] fileDropped -> load(" << p << ")";
                         this->load(p); // load() handles BusyScope + timings
                     });

    QObject::connect(m_view, &MainWindow::startOverRequested, m_view, [this](){
        qDebug() << "[CTRL] startOverRequested: clearing controller state + showing drag hint";
        this->onStartOverRequested();
    });


     io::dcmtk_global_init();
    // ---- Keyboard shortcuts for slice navigation ----
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

    // Start idle: show the drag hint (no gradient, no auto-load)
    m_view->beginNewImageCycle();
    qDebug() << "[CTRL] AppController ready (idle). Drag DICOM/HDF5 or use CLI path.";
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
// controller/app_controller.cpp
void AppController::clearLoadState()
{
    qDebug() << "[CTRL] clearLoadState()";
    m_meta.clear();
    m_probe = io::ProbeResult{};
    m_pre.clear(); m_preH = 0; m_preW = 0;
    m_display8.release();
    m_lastImg8.release();
    m_slices8.clear();
    m_currentSlice = 0;

    // IMPORTANT: do NOT call show_gradient_with_meta() here
    if (m_view) m_view->beginNewImageCycle();  // just show hint
}

void AppController::load(const QString& pathQ)
{
    qDebug() << "[CTRL][LOAD] enter pathQ=" << pathQ;

    if (m_view) m_view->beginNewImageCycle(); // clear UI state early

    if (pathQ.isEmpty()) {
        qWarning() << "[CTRL][LOAD] empty path; returning idle";
        return;
    }

    m_sourcePathQ = pathQ;
    const std::string path = pathQ.toStdString();
    bool anyWorkToShow = false;

    {   // -------- Busy only while probing/reading/reconstructing ----------
        BusyScope busy(m_view, QString("Loading %1").arg(pathQ));
        qDebug() << "[CTRL][LOAD] busy-enter";

        clearLoadState();
        m_meta << ("Source: " + pathQ);
        m_meta << ("When: " + QDateTime::currentDateTime().toString(Qt::ISODate));

        // Probe
        std::string dbg;
        QElapsedTimer tProbe; tProbe.start();
        m_probe = io::probe(path, &dbg);
        qDebug().noquote() << "[DBG][probe]\n" << QString::fromStdString(dbg);
        qDebug() << "[CTRL][LOAD] probe took ms=" << tProbe.elapsed();

        if (m_probe.flavor == io::Flavor::DICOM) {
            qDebug() << "[CTRL][LOAD] DICOM path -> load_dicom";
            QElapsedTimer t; t.start();
            const bool ok = load_dicom(path);
            qDebug() << "[CTRL][LOAD] load_dicom ok=" << ok << " ms=" << t.elapsed();
            anyWorkToShow |= ok;
        } else {
            qDebug() << "[CTRL][LOAD] HDF5 path -> reconstructAllSlicesFromDll";
            QElapsedTimer t; t.start();
            const bool ok = reconstructAllSlicesFromDll(pathQ, /*fftshift=*/true);
            qDebug() << "[CTRL][LOAD] DLL recon ok=" << ok << " ms=" << t.elapsed();
            if (!ok) {
                qWarning() << "[CTRL][LOAD] DLL recon failed -> prepare_fallback";
                prepare_fallback();
            }
            anyWorkToShow = true;
        }

        qDebug() << "[CTRL][LOAD] busy-leave";
    }   // BusyScope dtor here -> [View] endBusy should print BEFORE we paint

    if (!anyWorkToShow) {
        qWarning() << "[CTRL][LOAD] nothing to show; exit";
        return;
    }

    qDebug() << "[CTRL][LOAD] calling show() post-busy";
    show(); // only triggers lightweight UI work now
    qDebug() << "[CTRL][LOAD] exit";
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

#include <QElapsedTimer>
#include <cstdio>

// Keep ON until you see an image; then flip to 0 to re-enable multi-slice path.
#ifndef CTRL_DIAG_FORCE_FIRST_PAINT
#define CTRL_DIAG_FORCE_FIRST_PAINT 0
#endif

// AppController::load_dicom — simplified: use DLL read_all_gray8, split to cv::Mat, show
bool AppController::load_dicom(const std::string& path)
{
    qDebug() << "[DICOM][CTRL] load_dicom(enter) path=" << QString::fromStdString(path);

    // Reset view state early so the UI looks responsive
    if (m_view) {
        qDebug() << "[DICOM][CTRL] beginNewImageCycle() before decode";
        m_view->beginNewImageCycle();
        m_view->enableSliceSlider(0); // will be re-enabled properly after load
    }

    // Ensure the DLL wrapper exists and is loaded
    if (!m_dicom) m_dicom = std::make_unique<DicomDll>();
    if (!m_dicom->load()) {
        qWarning() << "[DICOM][CTRL][ERR] dicom__io_lib.dll not loaded";
        return false;
    }

    // (Optional) Query quick info — helpful for logs
    int infS=0, infW=0, infH=0, mono=0, spp=0, bits=0;
    if (m_dicom->info(path, &infS, &infW, &infH, &mono, &spp, &bits)) {
        qDebug() << "[DICOM][CTRL] info: " << infW << "x" << infH
                 << " frames=" << infS << " mono=" << mono
                 << " spp=" << spp  << " bits=" << bits;
    } else {
        qWarning() << "[DICOM][CTRL][WRN] dicom_info failed (continuing with read_all)";
    }

    // -------- Bulk read: all frames into a contiguous 8-bit stack --------
    QElapsedTimer t; t.start();
    uint8_t* stack = nullptr;
    int W = 0, H = 0, S = 0;

    qDebug() << "[DICOM][CTRL] calling DLL->read_all_gray8...";
    if (!m_dicom->read_all_gray8(path, &stack, &W, &H, &S) || !stack) {
        qWarning() << "[DICOM][CTRL][ERR] read_all_gray8 failed";
        if (m_view) { m_view->beginNewImageCycle(); m_view->enableSliceSlider(0); }
        return false;
    }
    qDebug() << "[DICOM][CTRL] AFTER DLL read-all" << "W=" << W << "H=" << H
             << "S=" << S << " ms=" << t.elapsed();

    // -------- Split stack into cv::Mat frames and fingerprint them --------
    std::vector<cv::Mat> frames;
    frames.reserve(std::max(S, 0));
    const size_t plane = static_cast<size_t>(W) * static_cast<size_t>(H);

    // cheap checksum helper to prove frames differ
    auto checksum16 = [&](const uint8_t* src)->qulonglong {
        if (!src || plane == 0) return 0;
        size_t step = plane / 16 + 1; // clarity over perf
        qulonglong cs = 0;
        for (size_t i = 0; i < plane; i += step) cs += src[i];
        return cs;
    };

    for (int f = 0; f < S; ++f) {
        const uint8_t* src = stack + plane * static_cast<size_t>(f);
        const qulonglong cs = checksum16(src);
        qDebug() << "[DICOM][CTRL] slice" << f << "checksum=" << cs;

        // Wrap then clone so controller owns pixels independent of DLL buffer
        cv::Mat u8(H, W, CV_8UC1, const_cast<uint8_t*>(src));
        frames.emplace_back(u8.clone());
    }
    // We no longer need the DLL buffer
    m_dicom->free_buf(stack);
    stack = nullptr;

    if (frames.empty()) {
        qWarning() << "[DICOM][CTRL] no frames after split";
        if (m_view) { m_view->beginNewImageCycle(); m_view->enableSliceSlider(0); }
        return false;
    }

// -------- Show in the UI --------
#if CTRL_DIAG_FORCE_FIRST_PAINT
    {
        qDebug() << "[DICOM][CTRL][DIAG] paint frame 0 dims=" << W << "x" << H
                 << " total=" << int(frames.size());
        m_meta.clear();
        m_meta << "Format: DICOM (diagnostic first-frame)"
               << QString("Dims: %1x%2").arg(W).arg(H)
               << QString("Frames: %1").arg(int(frames.size()));
        m_display8 = frames[0].clone();
        if (m_view) {
            m_view->setMetadata(m_meta);
            m_view->enableSliceSlider(0);
            m_view->setImage(m_display8);
        }
        return true;
    }
#else
    if (frames.size() >= 2) {
        qDebug() << "[DICOM][CTRL] multi-slice S=" << int(frames.size())
        << " dims=" << W << "x" << H;
        // Optional: log first few frame checksums again so View and Controller logs correlate
        for (int f = 0; f < std::min<int>(int(frames.size()), 4); ++f) {
            // compute mean as a second fingerprint
            cv::Scalar m = cv::mean(frames[f]);
            qDebug() << "[DICOM][CTRL][DIAG] frame" << f << "mean=" << m[0];
        }

        // your helper stores the stack + enables slider + paints slice 0
        showSlices(frames);
        return true;
    } else {
        qDebug() << "[DICOM][CTRL] single frame dims=" << W << "x" << H;
        m_meta.clear();
        m_meta << "Format: DICOM (via DLL)"
               << QString("Dims: %1x%2").arg(W).arg(H);
        m_display8 = frames[0].clone();
        if (m_view) {
            m_view->setMetadata(m_meta);
            m_view->enableSliceSlider(0);
            m_view->setImage(m_display8);
        }
        return true;
    }
#endif
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

// controller/app_controller.cpp
void AppController::prepare_fallback()
{
    qDebug() << "[CTRL] prepare_fallback(): creating gradient as a last resort";
    const int H = 512, W = 512;
    const cv::Mat grad = make_gradient(H, W);
    m_display8 = grad.clone();
    m_lastImg8 = grad.clone();

    QStringList meta;
    meta << "Source: <none>"
         << "Note: Fallback gradient (no input could be loaded)";
    show_metadata_and_image(meta, m_display8);
}


// ===============================
// Show pipeline
// ===============================
// controller/app_controller.cpp

// Helper: do we have anything to render?
static inline bool hasRenderable(const cv::Mat& single,
                                 const std::vector<cv::Mat>& stack) {
    return !single.empty() || !stack.empty();
}

void AppController::show()
{
    if (!m_view) return;

    if (!hasRenderable(m_display8, m_slices8)) {
        qDebug() << "[CTRL] show(): no content -> keep view idle (drag hint)";
        m_view->beginNewImageCycle();   // shows the drag hint
        return;
    }

    qDebug() << "[CTRL] show(): has content -> doShowNow()";
    doShowNow();
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

void AppController::showSlices(const std::vector<cv::Mat>& frames)
{
    qDebug() << "[DICOM][CTRL] showSlices ENTER  frames=" << (int)frames.size();
    m_slices8.clear();
    m_slices8.reserve(frames.size());

    int W = 0, H = 0;
    for (size_t i = 0; i < frames.size(); ++i) {
        const cv::Mat& f = frames[i];
        if (f.empty()) continue;

        if (i == 0) { W = f.cols; H = f.rows; }

        // normalize to 8-bit grayscale, deep copy for safety
        if (f.type() != CV_8UC1) {
            cv::Mat g;
            if (f.channels() == 3) cv::cvtColor(f, g, cv::COLOR_BGR2GRAY);
            else                    f.convertTo(g, CV_8U);
            m_slices8.emplace_back(g.clone());
        } else {
            m_slices8.emplace_back(f.clone());
        }
    }

    if (m_slices8.empty()) {
        qWarning() << "[CTRL] showSlices: all frames empty";
        return;
    }

    m_meta.clear();
    m_meta << "Format: DICOM (stack via DLL)"
           << QString("Dims: %1x%2").arg(W).arg(H);

    if (m_view) {
        m_view->setMetadata(m_meta);
        m_view->enableSliceSlider((int)m_slices8.size());
    }

    showSlice(0);
    qDebug() << "[DICOM][CTRL] showSlices EXIT (showed slice 0)";
}



void AppController::showSlice(int idx)
{
    const bool valid = !m_slices8.empty() && idx >= 0 && idx < int(m_slices8.size());
    qDebug() << "[CTRL] showSlice idx=" << idx << " valid=" << valid;
    if (!valid) {
        qWarning() << "[CTRL] showSlice: invalid; doShowNow()";
        doShowNow();
        return;
    }

    m_currentSlice = idx;
    if (m_view) m_view->setSliceIndex(idx); // lightweight

    QStringList meta = m_meta;
    meta << QString("Slice %1 / %2").arg(idx + 1).arg(m_slices8.size());


    // --- DIAGNOSTIC FINGERPRINT ---
    cv::Scalar mean = cv::mean(m_slices8[idx]);
    double l2prev = (idx > 0) ? cv::norm(m_slices8[idx], m_slices8[idx-1], cv::NORM_L2) : 0.0;
    qDebug() << "[View][DIAG] showing slice" << idx << " mean=" << mean[0] << " diff(prevL2)=" << l2prev;
    // --------------------------------

    show_metadata_and_image(meta, m_slices8[idx]); // calls m_view->setImage()
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

// In AppController
void AppController::onSliceChanged(int idx)
{
    qDebug() << "[CTRL] sliceChanged ->" << idx;
    showSlice(idx);   // <-- always route through this (don’t paint directly here)
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

AppController::BusyScope::BusyScope(MainWindow* v, const QString& message)
    : v_(v)
{
    qDebug() << "[BusyScope] begin:" << message;
    if (v_) v_->beginBusy(message);
}

AppController::BusyScope::~BusyScope()
{
    if (v_) v_->endBusy();
    qDebug() << "[BusyScope] end";
}

void AppController::onStartOverRequested()
{
    // Clear everything and re-enter idle (drag hint). clearLoadState() already calls beginNewImageCycle().
    clearLoadState();
    m_sourcePathQ.clear();
    m_meta.clear();
    qDebug() << "[CTRL] onStartOverRequested: state cleared; idle with drag hint";
}
