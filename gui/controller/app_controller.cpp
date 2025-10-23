// controller/app_controller.cpp
#include "app_controller.hpp"

#include "engine_api.h"

#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>

#include <QDateTime>
#include <QElapsedTimer>
#include <QShortcut>
#include <QDebug>
#include <QMetaObject>
#include <QCoreApplication>
#include <QFileInfo>
#include <QEventLoop>



#include "../view/mainwindow.hpp"
#include "../model/io.hpp"
#include "../model/dicom_dll.hpp"
#include "../src/image_utils.hpp"   // imgutil::to_u8_slice / make_test_gradient
#include "../view/progress_splash.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// =====================================================================
// AppController
// =====================================================================

// Engine → UI progress trampoline. Runs in engine thread or caller thread.
static void engine_progress_tramp(int pct, const char* stage, void* user)
{
    auto* self = reinterpret_cast<AppController*>(user);
    if (!self) return;

    const QString stageQ = QString::fromUtf8(stage ? stage : "");
    self->postSplashUpdateFromEngineThread(pct, stageQ);
}





AppController::AppController(MainWindow* view)
    : m_view(view)
{
    qDebug() << "[CTRL][ctor] AppController constructed; view?"
             << (m_view ? "YES" : "NO");

    // Register DCMTK codecs once for the process (safe to call multiple times)
    io::dcmtk_global_init();

    if (!m_view) {
        qWarning() << "[CTRL][ctor][WRN] MainWindow* is null; UI wiring skipped.";
        return;
    }

    // File saves
    QObject::connect(m_view, &MainWindow::requestSavePNG,  m_view,
                     [this](const QString& p){ qDebug() << "[CTRL] savePNG ->" << p; this->savePNG(p); });
    QObject::connect(m_view, &MainWindow::requestSaveDICOM, m_view,
                     [this](const QString& p){ qDebug() << "[CTRL] saveDICOM ->" << p; this->saveDICOM(p); });

    // Slice navigation (slider / wheel)
    QObject::connect(m_view, &MainWindow::sliceChanged, m_view,
                     [this](int idx){ qDebug() << "[CTRL] sliceChanged ->" << idx; this->onSliceChanged(idx); });

    // Drag and drop
    QObject::connect(m_view, &MainWindow::fileDropped, m_view,
                     [this](const QString& p){
                         qDebug() << "[DnD][CTRL] fileDropped -> load(" << p << ")";
                         this->load(p);
                     });



    // Start over (reset to idle hint)
    QObject::connect(m_view, &MainWindow::startOverRequested, m_view, [this](){
        qDebug() << "[CTRL] startOverRequested: clearing controller state + showing drag hint";
        this->onStartOverRequested();
    });

    // Keyboard shortcuts for slice navigation
    auto step = [this](int d){
        if (m_slices8.empty()) return;
        const int n = (int)m_slices8.size();
        showSlice(std::clamp(m_currentSlice + d, 0, n - 1));
    };
    auto jump = [this](int t){
        if (m_slices8.empty()) return;
        const int n = (int)m_slices8.size();
        showSlice(std::clamp(t, 0, n - 1));
    };
    auto bindStep = [&](const QKeySequence& ks, int d){
        auto* sc = new QShortcut(ks, m_view);
        QObject::connect(sc, &QShortcut::activated, m_view, [=]{ step(d); });
    };
    auto bindJump = [&](const QKeySequence& ks, int t){
        auto* sc = new QShortcut(ks, m_view);
        QObject::connect(sc, &QShortcut::activated, m_view, [=]{ jump(t); });
    };


    QObject::connect(m_view, &MainWindow::requestApplyNegative, m_view, [this](){
        qDebug() << "[CTRL][FX] requestApplyNegative -> toggleNegative()";
        this->toggleNegative();
    });


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

    // Idle state (drag hint)
    m_view->beginNewImageCycle();
    qDebug() << "[CTRL] AppController ready (idle). Drag DICOM/HDF5 or use CLI path.";
}

AppController::~AppController() = default;

// =====================================================================
// Small helpers
// =====================================================================

static inline QString ms(qint64 t) { return QString::number(t) + " ms"; }

cv::Mat AppController::to_u8(const cv::Mat& f32)
{
    if (f32.empty() || f32.type() != CV_32FC1) return {};
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
    cv::Mat f32(H, W, CV_32FC1, const_cast<float*>(v.data()));
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

// =====================================================================
// Load pipeline
// =====================================================================

void AppController::clearLoadState()
{
    qDebug() << "[CTRL] clearLoadState()";
    m_meta.clear();
    m_probe = io::ProbeResult{};
    m_display8.release();
    m_lastImg8.release();
    m_slices8.clear();
    m_currentSlice = 0;

    // --- Negative mode reset (ADD THIS BLOCK) ---
    m_negativeMode = false;
    m_slices8_base.clear();
    m_display8_base.release();
    if (m_view) m_view->onNegativeModeChanged(false);
    qDebug() << "[CTRL][FX] negative reset: OFF; caches cleared";

    if (m_view) m_view->beginNewImageCycle();  // show drag hint
}



void AppController::load(const QString& pathQ)
{
    qDebug() << "[CTRL][LOAD] enter pathQ=" << pathQ;
    if (m_view) m_view->beginNewImageCycle();

    if (pathQ.isEmpty()) {
        qWarning() << "[CTRL][LOAD] empty path; returning idle";
        return;
    }

    m_sourcePathQ = pathQ;
    const std::string path = pathQ.toStdString();
    bool anyWorkToShow = false;

    {   // Busy while probing/reading/reconstructing
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
        qDebug() << "[CTRL][LOAD] probe took" << tProbe.elapsed() << "ms";

        if (m_probe.flavor == io::Flavor::DICOM) {
            qDebug() << "[CTRL][LOAD] DICOM path -> load_dicom";
            QElapsedTimer t; t.start();
            const bool ok = load_dicom(path);
            qDebug() << "[CTRL][LOAD] load_dicom ok=" << ok << " ms=" << t.elapsed();
            if (!ok) {
                qWarning() << "[CTRL][LOAD] DICOM not admitted -> prepare_fallback";
                // m_meta was filled with the reason inside load_dicom()
                prepare_fallback();
                anyWorkToShow = true;  // we’ll show the fallback
            } else {
                anyWorkToShow = true;
            }
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
    }

    if (!anyWorkToShow) {
        qWarning() << "[CTRL][LOAD] nothing to show; exit";
        return;
    }

    qDebug() << "[CTRL][LOAD] calling show() post-busy";
    show();
    qDebug() << "[CTRL][LOAD] exit";
}


bool AppController::load_dicom(const std::string& path)
{
    qDebug() << "[DICOM][CTRL] load_dicom(enter) path=" << QString::fromStdString(path);

    // Reset view state early so the UI looks responsive
    if (m_view) {
        m_view->beginNewImageCycle();
        m_view->enableSliceSlider(0); // will be re-enabled after load
    }

    if (!m_dicom) m_dicom = std::make_unique<DicomDll>();
    if (!m_dicom->load()) {
        qWarning() << "[DICOM][CTRL][ERR] dicom__io_lib.dll not loaded";
        m_meta << "DICOM: loader not available (DLL missing)";
        return false;
    }

    // ---- INFO gate (admission policy) ----
    int frames=0, W=0, H=0, mono=0, spp=0, bits=0;
    {
        QElapsedTimer t; t.start();
        const bool okInfo = m_dicom->info(path, &frames, &W, &H, &mono, &spp, &bits);
        qDebug() << "[DICOM][CTRL] info() ok=" << okInfo
                 << " dims=" << W << "x" << H
                 << " frames=" << frames
                 << " mono=" << mono
                 << " spp=" << spp
                 << " bits=" << bits
                 << " ms=" << t.elapsed();

        if (!okInfo) {
            m_meta << "Format: DICOM (not admitted)"
                   << "Reason: could not read basic DICOM info (unsupported or corrupted)";
            return false;
        }
    }

    // Admission checks (clarity > perf)
    bool admitted = true;
    QStringList reasons;

    if (W <= 0 || H <= 0 || frames <= 0) {
        admitted = false; reasons << "invalid dimensions or zero frames";
    }
    if (mono != 1) {
        admitted = false; reasons << "requires MONOCHROME (Photometric MONOCHROME1/2)";
    }
    if (spp != 1) {
        admitted = false; reasons << "requires SamplesPerPixel=1";
    }
    if (!(bits == 8 || bits == 16)) {
        admitted = false; reasons << "requires BitsAllocated of 8 or 16";
    }

    // Memory sanity (avoid gigantic stacks)
    {
        const size_t plane = size_t(W) * size_t(H);
        const size_t bytes = plane * size_t(frames);
        const size_t kMaxStackBytes = 1024ull * 1024ull * 1024ull; // 1 GiB
        if (bytes == 0 || bytes > kMaxStackBytes) {
            admitted = false;
            reasons << QString("stack too large or zero (%1 bytes)").arg(qulonglong(bytes));
        }
    }

    if (!admitted) {
        qWarning() << "[DICOM][CTRL] NOT ADMITTED ->" << reasons.join("; ");
        m_meta.clear();
        m_meta << "Format: DICOM (not admitted)"
               << QString("Dims: %1x%2, Frames=%3").arg(W).arg(H).arg(frames)
               << QString("Mono=%1, SPP=%2, Bits=%3").arg(mono).arg(spp).arg(bits)
               << ("Reason: " + reasons.join("; "));
        return false; // caller will show graceful fallback
    }

    // ---- Read-all path (we only admit what we can show) ----
    QElapsedTimer t; t.start();
    uint8_t* stack = nullptr;
    int outW = 0, outH = 0, outS = 0;

    qDebug() << "[DICOM][CTRL] calling DLL->read_all_gray8...";
    if (!m_dicom->read_all_gray8(path, &stack, &outW, &outH, &outS) || !stack) {
        qWarning() << "[DICOM][CTRL][ERR] read_all_gray8 failed";
        m_meta.clear();
        m_meta << "Format: DICOM (not admitted)"
               << "Reason: failed to extract 8-bit frames for display";
        if (m_view) { m_view->beginNewImageCycle(); m_view->enableSliceSlider(0); }
        return false;
    }
    qDebug() << "[DICOM][CTRL] AFTER DLL read-all"
             << "W=" << outW << "H=" << outH
             << "S=" << outS << " ms=" << t.elapsed();

    // Split stack → cv::Mat frames (deep copies)
    std::vector<cv::Mat> framesU8;
    framesU8.reserve(std::max(outS, 0));
    const size_t plane = static_cast<size_t>(outW) * static_cast<size_t>(outH);

    auto checksum16 = [&](const uint8_t* src)->qulonglong {
        if (!src || plane == 0) return 0;
        size_t step = plane / 16 + 1; // clarity over perf
        qulonglong cs = 0;
        for (size_t i = 0; i < plane; i += step) cs += src[i];
        return cs;
    };

    for (int f = 0; f < outS; ++f) {
        const uint8_t* src = stack + plane * static_cast<size_t>(f);
        const qulonglong cs = checksum16(src);
        qDebug() << "[DICOM][CTRL] slice" << f << "checksum=" << cs;

        cv::Mat u8(outH, outW, CV_8UC1, const_cast<uint8_t*>(src));
        framesU8.emplace_back(u8.clone());     // own copy
    }

    // Release DLL buffer
    m_dicom->free_buf(stack);
    stack = nullptr;

    if (framesU8.empty()) {
        qWarning() << "[DICOM][CTRL] no frames after split";
        m_meta.clear();
        m_meta << "Format: DICOM (not admitted)"
               << "Reason: empty frame stack after read";
        if (m_view) { m_view->beginNewImageCycle(); m_view->enableSliceSlider(0); }
        return false;
    }

    // ---- Display path (single vs stack) ----
    if (framesU8.size() >= 2) {
        qDebug() << "[DICOM][CTRL] multi-slice S=" << int(framesU8.size())
        << " dims=" << outW << "x" << outH;
        showSlices(framesU8);
        m_meta.clear();
        m_meta << "Format: DICOM (admitted)"
               << QString("Dims: %1x%2, Frames=%3").arg(outW).arg(outH).arg(framesU8.size())
               << "Photometric: MONOCHROME (SPP=1)"
               << QString("BitsAllocated (src): %1").arg(bits);
    } else {
        qDebug() << "[DICOM][CTRL] single frame dims=" << outW << "x" << outH;
        m_meta.clear();
        m_meta << "Format: DICOM (admitted)"
               << QString("Dims: %1x%2").arg(outW).arg(outH)
               << "Photometric: MONOCHROME (SPP=1)"
               << QString("BitsAllocated (src): %1").arg(bits);
        m_display8 = framesU8[0].clone();
        if (m_view) {
            m_view->setMetadata(m_meta);
            m_view->enableSliceSlider(0);
            m_view->setImage(m_display8);
        }
    }
    return true;
}


// HDF5/ISMRMRD reconstruction via engine DLL with Splash progress
bool AppController::reconstructAllSlicesFromDll(const QString& pathQ, bool fftshift)
{
    qDebug() << "[DBG][DLL] reconstructAllSlicesFromDll path=" << pathQ << " fftshift=" << fftshift;

    // --- Show splash at start -------------------------------------------------
    if (!m_splash) m_splash = new ProgressSplash(m_view);
    const QString title = QStringLiteral("Reconstructing");
    m_splash->start(title);

    // --- Initialize engine once ----------------------------------------------
    static std::once_flag s_once;
    static int s_init_ok = 0;
    std::call_once(s_once, [&](){
        const char* ver = engine_version();
        qDebug() << "[DBG][DLL] engine_version ->" << (ver ? ver : "(null)");
        s_init_ok = engine_init(0);  // 0=auto/best, -1=force CPU
        qDebug() << "[DBG][DLL] engine_init ->" << s_init_ok;
    });
    if (!s_init_ok) {
        qWarning() << "[DBG][DLL] engine_init failed; skipping DLL path";
        closeSplashIfAny();
        return false;
    }

    // --- Hook engine progress -> splash --------------------------------------
    engine_set_progress_cb(&engine_progress_tramp, this);

    // --- Call engine ----------------------------------------------------------
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

    // --- Unhook progress ASAP -------------------------------------------------
    engine_set_progress_cb(nullptr, nullptr);

    if (dbg[0] != '\0') {
        qDebug().noquote() << "[DBG][DLL] " << dbg;
    }

    if (!ok || !stack || S <= 0 || H <= 0 || W <= 0) {
        qWarning() << "[DBG][DLL] reconstruct_all failed or invalid dims"
                   << " ok=" << ok << " stack=" << (void*)stack
                   << " S=" << S << " H=" << H << " W=" << W;
        if (stack) engine_free(stack);
        closeSplashIfAny();
        return false;
    }

    // --- Success: copy to host, free engine buffer ---------------------------
    const size_t count = size_t(S) * size_t(H) * size_t(W);
    std::vector<float> host(count);
    std::memcpy(host.data(), stack, count * sizeof(float));
    engine_free(stack);

    // Convert to 8-bit stack and enable slider
    adoptReconStackF32(host, S, H, W);
    m_meta << QString("DLL: Slices=%1, Size=%2x%3").arg(S).arg(W).arg(H);

    // Optional: extract ISMRMRD metadata for UI (unchanged from your version)
    {
        io::DicomMeta dm;
        std::string whyMeta;
        const std::string p = pathQ.toStdString();
        if (io::read_hdf5_ismrmrd_meta(p, dm, &whyMeta)) {
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
        } else if (!whyMeta.empty()) {
            qWarning() << "[DBG][H5][META] read failed:" << QString::fromStdString(whyMeta);
        }
    }

    qDebug() << "[DBG][DLL] reconstructAllSlicesFromDll OK: S=" << S << " H=" << H << " W=" << W;

    if (m_splash) m_splash->updateProgress(100, "Done");
    closeSplashIfAny();
    return true;
}


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

// =====================================================================
// Show pipeline
// =====================================================================

static inline bool hasRenderable(const cv::Mat& single,
                                 const std::vector<cv::Mat>& stack) {
    return !single.empty() || !stack.empty();
}

void AppController::show()
{
    if (!m_view) return;

    if (!hasRenderable(m_display8, m_slices8)) {
        qDebug() << "[CTRL] show(): no content -> keep view idle (drag hint)";
        m_view->beginNewImageCycle();
        return;
    }
    qDebug() << "[CTRL] show(): has content -> doShowNow()";
    doShowNow();
}

void AppController::doShowNow()
{
    if (!m_view) return;

    if (!m_slices8.empty()) {
        m_view->enableSliceSlider((int)m_slices8.size());
        showSlice(std::clamp(m_currentSlice, 0, (int)m_slices8.size()-1));
        return;
    }

    if (m_display8.empty() || m_display8.type() != CV_8UC1) {
        QStringList meta = m_meta; meta << "Display: forced gradient (no valid 8-bit image)";
        show_gradient_with_meta(meta);
        return;
    }

    m_view->enableSliceSlider(0);
    show_metadata_and_image(m_meta, m_display8);
}

void AppController::show_metadata_and_image(const QStringList& meta, const cv::Mat& u8)
{
    if (!m_view) return;
    m_view->setMetadata(meta);
    m_view->setImage(u8);
    m_lastImg8 = u8.clone();
}

void AppController::show_gradient_with_meta(const QStringList& meta)
{
    if (!m_view) return;
    cv::Mat grad = make_gradient(512, 512);
    m_view->setMetadata(meta);
    m_view->setImage(grad);
    m_lastImg8 = grad.clone();
}

// =====================================================================
// Multi-slice helpers
// =====================================================================

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

        // Normalize to 8-bit grayscale (defensive)
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

    // Diagnostics
    cv::Scalar mean = cv::mean(m_slices8[idx]);
    double l2prev = (idx > 0) ? cv::norm(m_slices8[idx], m_slices8[idx-1], cv::NORM_L2) : 0.0;
    qDebug() << "[View][DIAG] showing slice" << idx << " mean=" << mean[0] << " diff(prevL2)=" << l2prev;

    show_metadata_and_image(meta, m_slices8[idx]);
}

// =====================================================================
// Save operations
// =====================================================================

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

// =====================================================================
// View callbacks
// =====================================================================

void AppController::onSliceChanged(int idx)
{
    qDebug() << "[CTRL] sliceChanged ->" << idx;
    showSlice(idx);
}

void AppController::onStartOverRequested()
{
    qDebug() << "[CTRL] onStartOverRequested ENTER";
    clearLoadState();
    m_sourcePathQ.clear();
    m_meta.clear();

    if (m_view) {
        qDebug() << "[CTRL] onStartOverRequested: calling view->beginNewImageCycle()";
        m_view->beginNewImageCycle();  // clears pixmap + disables slider (now without emitting)
    }

    qDebug() << "[CTRL] onStartOverRequested: state cleared; idle with drag hint";
}

// =====================================================================
// BusyScope (UI busy indicator)
// =====================================================================

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


// app_controller.cpp

void AppController::applyNegative()
{
    qDebug() << "[CTRL][FX] applyNegative ENTER";

    auto invertMat = [](const cv::Mat& src)->cv::Mat {
        if (src.empty()) return {};
        cv::Mat src8, dst;
        if (src.type() != CV_8UC1) {
            qDebug() << "[CTRL][FX] src not 8UC1, converting";
            src.convertTo(src8, CV_8U);
        } else {
            src8 = src;
        }
        double mn=0.0, mx=0.0; cv::minMaxLoc(src8, &mn, &mx);
        qDebug() << "[CTRL][FX] before invert: min=" << mn << " max=" << mx
                 << " size=" << src8.cols << "x" << src8.rows;
        cv::bitwise_not(src8, dst);  // 8-bit negative
        cv::minMaxLoc(dst, &mn, &mx);
        qDebug() << "[CTRL][FX] after  invert: min=" << mn << " max=" << mx;
        return dst;
    };

    if (!m_slices8.empty()) {
        const bool idxOk = (m_currentSlice >= 0 && m_currentSlice < (int)m_slices8.size());
        if (!idxOk || m_slices8[m_currentSlice].empty()) {
            qWarning() << "[CTRL][FX] multi-slice but current slice invalid/empty";
            return;
        }

        qDebug() << "[CTRL][FX] multi-slice negative on slice" << m_currentSlice
                 << " of " << m_slices8.size();
        cv::Mat neg = invertMat(m_slices8[m_currentSlice]);
        if (neg.empty()) { qWarning() << "[CTRL][FX] invert returned empty"; return; }

        m_slices8[m_currentSlice] = neg;  // replace in-place

        // Optional: annotate metadata (non-destructive; shown via overlay)
        if (m_view) m_view->appendMetadataLine(
                QString("Effect: Negative applied to slice %1").arg(m_currentSlice + 1));

        // This will refresh view and also set m_lastImg8 via show_metadata_and_image()
        showSlice(m_currentSlice);

    } else if (!m_display8.empty()) {
        qDebug() << "[CTRL][FX] single-image negative";
        cv::Mat neg = invertMat(m_display8);
        if (neg.empty()) { qWarning() << "[CTRL][FX] invert returned empty"; return; }

        m_display8 = neg;

        // Optionally persist a marker in the current metadata
        QStringList meta = m_meta;
        meta << "Effect: Negative";

        // This updates the UI and also updates m_lastImg8 used by savePNG/saveDICOM
        show_metadata_and_image(meta, m_display8);
    } else {
        qWarning() << "[CTRL][FX] nothing to invert (no display image / slices)";
        return;
    }

    qDebug() << "[CTRL][FX] applyNegative EXIT (image updated; saving will use new m_lastImg8)";
}


cv::Mat AppController::invert8u(const cv::Mat& src)
{
    if (src.empty()) return {};
    cv::Mat src8, dst;
    if (src.type() != CV_8UC1) {
        qDebug() << "[CTRL][FX] invert8u: converting type" << src.type() << "-> 8UC1";
        src.convertTo(src8, CV_8U);
    } else {
        src8 = src;
    }
    double mn=0.0, mx=0.0; cv::minMaxLoc(src8, &mn, &mx);
    qDebug() << "[CTRL][FX] invert8u: before min=" << mn << " max=" << mx
             << " size=" << src8.cols << "x" << src8.rows;
    cv::bitwise_not(src8, dst);
    cv::minMaxLoc(dst, &mn, &mx);
    qDebug() << "[CTRL][FX] invert8u: after  min=" << mn << " max=" << mx;
    return dst;
}


void AppController::captureNegativeBaseIfNeeded()
{
    // Lazy snapshot the first time we go ON
    if (!m_slices8_base.empty() || !m_display8_base.empty()) return;

    if (!m_slices8.empty()) {
        qDebug() << "[CTRL][FX] capture base for negative (multi-slice)"
                 << "count=" << (int)m_slices8.size();
        m_slices8_base.clear();
        m_slices8_base.reserve(m_slices8.size());
        for (const auto& s : m_slices8) m_slices8_base.push_back(s.clone());
    } else if (!m_display8.empty()) {
        qDebug() << "[CTRL][FX] capture base for negative (single image)";
        m_display8_base = m_display8.clone();
    } else {
        qDebug() << "[CTRL][FX] capture base skipped: no image loaded";
    }
}


void AppController::toggleNegative()
{
    qDebug() << "[CTRL][FX] toggleNegative ENTER, current state =" << (m_negativeMode ? "ON" : "OFF");

    if (!m_negativeMode) captureNegativeBaseIfNeeded(); // snapshot before turning ON
    m_negativeMode = !m_negativeMode;

    if (m_negativeMode) {
        // ON: invert ALL slices or the single image, based on the base snapshot
        if (!m_slices8_base.empty()) {
            qDebug() << "[CTRL][FX] Negative ON: inverting all slices";
            m_slices8.clear();
            m_slices8.reserve(m_slices8_base.size());
            for (size_t i = 0; i < m_slices8_base.size(); ++i) {
                m_slices8.push_back(invert8u(m_slices8_base[i]));
            }
            int target = (m_currentSlice >= 0 && m_currentSlice < (int)m_slices8.size())
                             ? m_currentSlice : 0;
            if (m_view) m_view->appendMetadataLine("View Mode: Negative (all slices)");
            showSlice(target);  // refresh UI + m_lastImg8 used by saving

        } else if (!m_display8_base.empty()) {
            qDebug() << "[CTRL][FX] Negative ON: inverting single image";
            m_display8 = invert8u(m_display8_base);
            QStringList meta = m_meta; meta << "View Mode: Negative";
            show_metadata_and_image(meta, m_display8);

        } else {
            qWarning() << "[CTRL][FX] Negative ON requested but no base image is present";
        }

    } else {
        // OFF: restore base snapshot (non-negative)
        if (!m_slices8_base.empty()) {
            qDebug() << "[CTRL][FX] Negative OFF: restoring base slices";
            m_slices8.clear();
            m_slices8.reserve(m_slices8_base.size());
            for (const auto& b : m_slices8_base) m_slices8.push_back(b.clone());
            int target = (m_currentSlice >= 0 && m_currentSlice < (int)m_slices8.size())
                             ? m_currentSlice : 0;
            if (m_view) m_view->appendMetadataLine("View Mode: Normal");
            showSlice(target);

        } else if (!m_display8_base.empty()) {
            qDebug() << "[CTRL][FX] Negative OFF: restoring base image";
            m_display8 = m_display8_base.clone();
            QStringList meta = m_meta; meta << "View Mode: Normal";
            show_metadata_and_image(meta, m_display8);
        }
    }

    if (m_view) m_view->onNegativeModeChanged(m_negativeMode); // keep menu check in sync
    qDebug() << "[CTRL][FX] toggleNegative EXIT, new state =" << (m_negativeMode ? "ON" : "OFF");
}

void AppController::closeSplashIfAny()
{
    if (m_splash) {
        qDebug() << "[CTRL][SPLASH] closing";
        m_splash->finish();
        m_splash = nullptr;
    }
}

void AppController::postSplashUpdateFromEngineThread(int pct, const QString& stage)
{
    qDebug() << "[CTRL][SPLASH] postSplashUpdateFromEngineThread pct=" << pct << " stage=" << stage;

    if (m_splash) {
        QMetaObject::invokeMethod(
            m_splash,
            [sp=m_splash, pct, stage](){
                if (sp) {
                    sp->updateProgress(pct, stage);
                    qDebug() << "[CTRL][SPLASH] invoke -> updateProgress" << pct << stage;
                } else {
                    qDebug() << "[CTRL][SPLASH] invoke skipped (splash gone)";
                }
            },
            Qt::QueuedConnection
            );
    } else {
        qDebug() << "[CTRL][SPLASH] no splash; skipping update";
    }

    // If engine is called synchronously on GUI thread, allow paints
    QCoreApplication::processEvents(QEventLoop::AllEvents, 8);
}

