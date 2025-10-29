#include "app_controller.hpp"
#include "engine_api.h"
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>
#include <QApplication>
#include <QMetaObject>
#include <QDateTime>
#include <QElapsedTimer>
#include <QShortcut>
#include <QDebug>
#include <QCoreApplication>
#include <QFileInfo>
#include <QEventLoop>
#include <QTimer>
#include "../view/mainwindow.hpp"
#include "../model/io.hpp"
#include "../model/dicom_dll.hpp"
#include "../src/image_utils.hpp"
#include "../view/progress_splash.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

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
    qDebug() << "[CTRL][ctor] AppController constructed; view?" << (m_view ? "YES" : "NO");
    io::dcmtk_global_init();

    if (!m_view) {
        qWarning() << "[CTRL][ctor][WRN] MainWindow* is null; UI wiring skipped.";
        return;
    }

    QObject::connect(m_view, &MainWindow::requestSavePNG,  m_view,
                     [this](const QString& p){ qDebug() << "[CTRL] savePNG ->" << p; this->savePNG(p); });
    QObject::connect(m_view, &MainWindow::requestSaveDICOM, m_view,
                     [this](const QString& p){ qDebug() << "[CTRL] saveDICOM ->" << p; this->saveDICOM(p); });

    QObject::connect(m_view, &MainWindow::sliceChanged, m_view,
                     [this](int idx){ qDebug() << "[CTRL] sliceChanged ->" << idx; this->onSliceChanged(idx); });

    QObject::connect(m_view, &MainWindow::fileDropped, m_view,
                     [this](const QString& p){
                         qDebug() << "[DnD][CTRL] fileDropped -> load(" << p << ")";
                         this->load(p);
                     });

    QObject::connect(m_view, &MainWindow::startOverRequested, m_view, [this](){
        qDebug() << "[CTRL] startOverRequested: clearing controller state + showing drag hint";
        this->onStartOverRequested();
    });

    QObject::connect(m_view, &MainWindow::requestHistogramUpdate, m_view,
                     [this](const QSize& s){
                         qDebug() << "[CTRL][Hist] requestHistogramUpdate -> onHistogramUpdateRequested size=" << s;
                         this->onHistogramUpdateRequested(s);
                     });

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

    m_view->beginNewImageCycle();
    qDebug() << "[CTRL] AppController ready (idle). Drag DICOM/HDF5 or use CLI path.";
}

AppController::~AppController() = default;

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

void AppController::clearLoadState()
{
    qDebug() << "[CTRL] clearLoadState()";
    m_meta.clear();
    m_probe = io::ProbeResult{};
    m_display8.release();
    m_lastImg8.release();
    m_slices8.clear();
    m_currentSlice = 0;
    m_negativeMode = false;
    m_slices8_base.clear();
    m_display8_base.release();
    if (m_view) m_view->onNegativeModeChanged(false);
    qDebug() << "[CTRL][FX] negative reset: OFF; caches cleared";
    if (m_view) m_view->beginNewImageCycle();
}

void AppController::load(const QString& pathQ)
{
    qDebug() << "[CTRL][LOAD] enter pathQ=" << pathQ;
    if (m_view) m_view->beginNewImageCycle();
    if (pathQ.isEmpty()) { qWarning() << "[CTRL][LOAD] empty path; returning idle"; return; }

    m_sourcePathQ = pathQ;
    bool anyWorkToShow = false;

    {   BusyScope busy(m_view, QString("Loading %1").arg(pathQ));
        qDebug() << "[CTRL][LOAD] busy-enter";

        clearLoadState();
        m_meta << ("Source: " + pathQ);

        std::string dbg;
        QElapsedTimer tProbe; tProbe.start();
        m_probe = io::probe(pathQ.toStdString(), &dbg);
        qDebug().noquote() << "[DBG][probe]\n" << QString::fromStdString(dbg);
        qDebug() << "[CTRL][LOAD] probe took" << tProbe.elapsed() << "ms";

        if (m_probe.flavor == io::Flavor::DICOM) {
            qDebug() << "[CTRL][LOAD] DICOM path -> loadDicom";
            QElapsedTimer t; t.start();
            const bool ok = this->loadDicom(pathQ);
            qDebug() << "[CTRL][LOAD] loadDicom ok=" << ok << " ms=" << t.elapsed();
            if (!ok) { qWarning() << "[CTRL][LOAD] DICOM not admitted -> prepare_fallback"; prepare_fallback(); anyWorkToShow = true; }
            else      anyWorkToShow = true;
        } else {
            qDebug() << "[CTRL][LOAD] HDF5 path -> reconstructAllSlicesFromDll";
            QElapsedTimer t; t.start();
            const bool ok = reconstructAllSlicesFromDll(pathQ, /*fftshift=*/true);
            qDebug() << "[CTRL][LOAD] DLL recon ok=" << ok << " ms=" << t.elapsed();
            if (!ok) { qWarning() << "[CTRL][LOAD] DLL recon failed -> prepare_fallback"; prepare_fallback(); }
            anyWorkToShow = true;
        }

        qDebug() << "[CTRL][LOAD] busy-leave";
    }

    if (!anyWorkToShow) { qWarning() << "[CTRL][LOAD] nothing to show; exit"; return; }
    qDebug() << "[CTRL][LOAD] calling show() post-busy";
    show();
    qDebug() << "[CTRL][LOAD] exit";
}

bool AppController::reconstructAllSlicesFromDll(const QString& pathQ, bool fftshift)
{
    qDebug() << "[DBG][DLL] reconstructAllSlicesFromDll path=" << pathQ << " fftshift=" << fftshift;
    if (!m_splash) m_splash = new ProgressSplash(m_view);
    const QString title = QStringLiteral("Reconstructing");
    m_splash->start(title);

    static std::once_flag s_once;
    static int s_init_ok = 0;
    std::call_once(s_once, [&](){
        const char* ver = engine_version();
        qDebug() << "[DBG][DLL] engine_version ->" << (ver ? ver : "(null)");
        s_init_ok = engine_init(0);
        qDebug() << "[DBG][DLL] engine_init ->" << s_init_ok;
    });
    if (!s_init_ok) { qWarning() << "[DBG][DLL] engine_init failed; skipping DLL path"; closeSplashIfAny(); return false; }

    engine_set_progress_cb(&engine_progress_tramp, this);

    // --- Call engine ----------------------------------------------------------
    int S = 0, H = 0, W = 0;
    float* stack = nullptr;
    char dbg[4096] = {0};
    const QByteArray path8 = pathQ.toUtf8();
    const int ok = engine_reconstruct_all(
        path8.constData(), &S, &H, &W, &stack, fftshift ? 1 : 0, dbg, int(sizeof(dbg)));

    engine_set_progress_cb(nullptr, nullptr);

    if (dbg[0] != '\0') qDebug().noquote() << "[DBG][DLL] " << dbg;

    if (!ok || !stack || S <= 0 || H <= 0 || W <= 0) {
        qWarning() << "[DBG][DLL] reconstruct_all failed or invalid dims"
                   << " ok=" << ok << " stack=" << (void*)stack
                   << " S=" << S << " H=" << H << " W=" << W;
        if (stack) engine_free(stack);
        closeSplashIfAny();
        return false;
    }

    const size_t count = size_t(S) * size_t(H) * size_t(W);
    std::vector<float> host(count);
    std::memcpy(host.data(), stack, count * sizeof(float));
    engine_free(stack);

    adoptReconStackF32(host, S, H, W);
    m_meta << QString("DLL: Slices=%1, Size=%2x%3").arg(S).arg(W).arg(H);

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

static inline bool hasRenderable(const cv::Mat& single, const std::vector<cv::Mat>& stack) {
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
        if (f.type() != CV_8UC1) {
            cv::Mat g;
            if (f.channels() == 3) cv::cvtColor(f, g, cv::COLOR_BGR2GRAY);
            else                    f.convertTo(g, CV_8U);
            m_slices8.emplace_back(g.clone());
        } else {
            m_slices8.emplace_back(f.clone());
        }
    }

    if (m_slices8.empty()) { qWarning() << "[CTRL] showSlices: all frames empty"; return; }

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
    if (!valid) { qWarning() << "[CTRL] showSlice: invalid; doShowNow()"; doShowNow(); return; }

    m_currentSlice = idx;
    if (m_view) m_view->setSliceIndex(idx);
    QStringList meta = m_meta;
    meta << QString("Slice %1 / %2").arg(idx + 1).arg(m_slices8.size());
    cv::Scalar mean = cv::mean(m_slices8[idx]);
    double l2prev = (idx > 0) ? cv::norm(m_slices8[idx], m_slices8[idx-1], cv::NORM_L2) : 0.0;
    qDebug() << "[View][DIAG] showing slice" << idx << " mean=" << mean[0] << " diff(prevL2)=" << l2prev;
    show_metadata_and_image(meta, m_slices8[idx]);
}

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
        m_view->beginNewImageCycle();
    }
    qDebug() << "[CTRL] onStartOverRequested: state cleared; idle with drag hint";
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

QImage AppController::renderHistogram(const cv::Mat& srcRef,
                                      const QSize& canvasSize,
                                      bool negativeMode,
                                      QString* tooltip)
{
    const int W = std::max(100, canvasSize.width());
    const int H = std::max(80,  canvasSize.height());
    const cv::Scalar bg   = negativeMode ? cv::Scalar(0,   0,   0)   : cv::Scalar(255, 255, 255);
    const cv::Scalar bar  = cv::Scalar(255, 0,   0);
    const cv::Scalar axis = negativeMode ? cv::Scalar(90,  90,  90)  : cv::Scalar(180, 180, 180);

    cv::Mat canvas(H, W, CV_8UC3, bg);
    auto drawBaseline = [&](){ cv::line(canvas, cv::Point(0, H - 1), cv::Point(W - 1, H - 1), axis, 1, cv::LINE_8); };

    if (srcRef.empty()) {
        drawBaseline();
        const std::string text = "No image";
        int baseline = 0;
        cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
        cv::Point org((W - ts.width)/2, (H + ts.height)/2);
        cv::putText(canvas, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    negativeMode ? cv::Scalar(200,200,200) : cv::Scalar(50,50,50), 1, cv::LINE_AA);
        if (tooltip) *tooltip = QString("Histogram: no image");
        QImage qi(canvas.data, canvas.cols, canvas.rows, int(canvas.step), QImage::Format_BGR888);
        return qi.copy();
    }

    cv::Mat src8;
    if (srcRef.type() == CV_8UC1) src8 = srcRef;
    else                          srcRef.convertTo(src8, CV_8U);

    constexpr int histSize = 256;
    float range[2] = {0.f, 256.f};
    const float* ranges[] = { range };
    int channels[] = { 0 };
    cv::Mat hist;
    cv::calcHist(&src8, 1, channels, cv::Mat(), hist, 1, &histSize, ranges, true, false);

    {
        const int start = negativeMode ? std::max(0, hist.rows - 5) : 0;
        const int end   = negativeMode ? hist.rows - 1              : std::min(hist.rows - 1, 4);
        double suppressed = 0.0;
        for (int i = start; i <= end; ++i) { suppressed += hist.at<float>(i); hist.at<float>(i) = 0.f; }
        qDebug() << "[CTRL][Hist] background bins ignored:" << start << "..." << end << " suppressed=" << suppressed;
    }

    double minv = 0.0, maxv = 0.0;
    cv::minMaxLoc(hist, &minv, &maxv);

    const int topMargin   = 12;
    const int leftMargin  = 6;
    const int rightMargin = 6;
    const int usableW     = std::max(1, W - leftMargin - rightMargin);
    const int baseY       = H - 1;

    drawBaseline();

    const double hScale = (maxv > 0.0) ? (double(H - topMargin - 1) / maxv) : 0.0;
    const double binWf  = double(usableW) / double(histSize);

    for (int i = 0; i < histSize; ++i) {
        const int x0 = leftMargin + std::clamp(int(std::floor(i * binWf)),  0, std::max(0, W - 1));
        int       x1 = leftMargin + std::clamp(int(std::floor((i + 1) * binWf)) - 1, 0, std::max(0, W - 1));
        if (x1 < x0) x1 = x0;
        int h = int(std::round(hist.at<float>(i) * hScale));
        h = std::clamp(h, 0, std::max(0, H - topMargin));
        cv::rectangle(canvas, {x0, baseY}, {x1, baseY - h}, bar, cv::FILLED, cv::LINE_8);
    }

    QImage qi(canvas.data, canvas.cols, canvas.rows, int(canvas.step), QImage::Format_BGR888);
    if (tooltip) {
        *tooltip = QString("Histogram: 256 bins | bg ignored: %1 | size=%2x%3 | canvas=%4x%5")
        .arg(negativeMode ? "251..255" : "0..4")
            .arg(src8.cols).arg(src8.rows)
            .arg(W).arg(H);
    }
    return qi.copy();
}

void AppController::onHistogramUpdateRequested(const QSize& canvas)
{
    qDebug() << "[CTRL][Hist] onHistogramUpdateRequested ENTER size=" << canvas;
    if (!m_view) { qWarning() << "[CTRL][Hist] no view"; return; }

    // Pick the best available 8-bit image: prefer last shown; else current single; else current slice
    cv::Mat src;
    if (!m_lastImg8.empty()) {
        src = m_lastImg8;
    } else if (!m_display8.empty()) {
        src = m_display8;
    } else if (!m_slices8.empty()) {
        const int idx = (m_currentSlice >= 0 && m_currentSlice < (int)m_slices8.size()) ? m_currentSlice : 0;
        src = m_slices8[idx];
    }

    QString tip;
    QImage histImg = renderHistogram(src, canvas, m_negativeMode, &tip);
    m_view->setHistogramImage(histImg, tip);
    qDebug() << "[CTRL][Hist] onHistogramUpdateRequested EXIT";
}

void AppController::applyNegative()
{
    qDebug() << "[CTRL][FX] applyNegative ENTER";

    auto invertMat = [](const cv::Mat& src)->cv::Mat {
        if (src.empty()) return {};
        cv::Mat src8, dst;
        if (src.type() != CV_8UC1) { src.convertTo(src8, CV_8U); } else { src8 = src; }
        double mn=0.0, mx=0.0; cv::minMaxLoc(src8, &mn, &mx);
        qDebug() << "[CTRL][FX] before invert: min=" << mn << " max=" << mx
                 << " size=" << src8.cols << "x" << src8.rows;
        cv::bitwise_not(src8, dst);
        cv::minMaxLoc(dst, &mn, &mx);
        qDebug() << "[CTRL][FX] after  invert: min=" << mn << " max=" << mx;
        return dst;
    };

    if (!m_slices8.empty()) {
        const bool idxOk = (m_currentSlice >= 0 && m_currentSlice < (int)m_slices8.size());
        if (!idxOk || m_slices8[m_currentSlice].empty()) { qWarning() << "[CTRL][FX] multi-slice but current slice invalid/empty"; return; }
        qDebug() << "[CTRL][FX] multi-slice negative on slice" << m_currentSlice << " of " << m_slices8.size();
        cv::Mat neg = invertMat(m_slices8[m_currentSlice]);
        if (neg.empty()) { qWarning() << "[CTRL][FX] invert returned empty"; return; }
        m_slices8[m_currentSlice] = neg;
        if (m_view) m_view->appendMetadataLine(QString("Effect: Negative applied to slice %1").arg(m_currentSlice + 1));
        showSlice(m_currentSlice);
    } else if (!m_display8.empty()) {
        qDebug() << "[CTRL][FX] single-image negative";
        cv::Mat neg = invertMat(m_display8);
        if (neg.empty()) { qWarning() << "[CTRL][FX] invert returned empty"; return; }
        m_display8 = neg;
        QStringList meta = m_meta;
        meta << "Effect: Negative";
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
    if (src.type() != CV_8UC1) { qDebug() << "[CTRL][FX] invert8u: converting type" << src.type() << "-> 8UC1"; src.convertTo(src8, CV_8U); }
    else { src8 = src; }
    double mn=0.0, mx=0.0; cv::minMaxLoc(src8, &mn, &mx);
    qDebug() << "[CTRL][FX] invert8u: before min=" << mn << " max=" << mx << " size=" << src8.cols << "x" << src8.rows;
    cv::bitwise_not(src8, dst);
    cv::minMaxLoc(dst, &mn, &mx);
    qDebug() << "[CTRL][FX] invert8u: after  min=" << mn << " max=" << mx;
    return dst;
}

void AppController::captureNegativeBaseIfNeeded()
{
    if (!m_slices8_base.empty() || !m_display8_base.empty()) return;
    if (!m_slices8.empty()) {
        qDebug() << "[CTRL][FX] capture base for negative (multi-slice)" << "count=" << (int)m_slices8.size();
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
    if (!m_negativeMode) captureNegativeBaseIfNeeded();
    m_negativeMode = !m_negativeMode;

    if (m_negativeMode) {
        if (!m_slices8_base.empty()) {
            qDebug() << "[CTRL][FX] Negative ON: inverting all slices";
            m_slices8.clear();
            m_slices8.reserve(m_slices8_base.size());
            for (size_t i = 0; i < m_slices8_base.size(); ++i) m_slices8.push_back(invert8u(m_slices8_base[i]));
            int target = (m_currentSlice >= 0 && m_currentSlice < (int)m_slices8.size()) ? m_currentSlice : 0;
            if (m_view) m_view->appendMetadataLine("View Mode: Negative (all slices)");
            showSlice(target);
        } else if (!m_display8_base.empty()) {
            qDebug() << "[CTRL][FX] Negative ON: inverting single image";
            m_display8 = invert8u(m_display8_base);
            QStringList meta = m_meta; meta << "View Mode: Negative";
            show_metadata_and_image(meta, m_display8);
        } else {
            qWarning() << "[CTRL][FX] Negative ON requested but no base image is present";
        }
    } else {
        if (!m_slices8_base.empty()) {
            qDebug() << "[CTRL][FX] Negative OFF: restoring base slices";
            m_slices8.clear();
            m_slices8.reserve(m_slices8_base.size());
            for (const auto& b : m_slices8_base) m_slices8.push_back(b.clone());
            int target = (m_currentSlice >= 0 && m_currentSlice < (int)m_slices8.size()) ? m_currentSlice : 0;
            if (m_view) m_view->appendMetadataLine("View Mode: Normal");
            showSlice(target);
        } else if (!m_display8_base.empty()) {
            qDebug() << "[CTRL][FX] Negative OFF: restoring base image";
            m_display8 = m_display8_base.clone();
            QStringList meta = m_meta; meta << "View Mode: Normal";
            show_metadata_and_image(meta, m_display8);
        }
    }

    if (m_view) {
        m_view->onNegativeModeChanged(m_negativeMode);
    }
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
    QCoreApplication::processEvents(QEventLoop::AllEvents, 8);
}

bool AppController::loadDicom(const QString& pathUtf8)
{
    qDebug() << "[CTRL][loadDicom] ENTER path=" << pathUtf8;
    const std::string path = pathUtf8.toStdString();

    std::vector<cv::Mat> frames8;
    std::string why;

    if (!decodeDicomToFrames8(path, frames8, why) || frames8.empty()) {
        qWarning() << "[CTRL][loadDicom][WRN] decode failed:" << QString::fromStdString(why);
        qDebug() << "[CTRL][loadDicom] EXIT fail";
        return false;
    }

    adoptFrames8ToState(frames8);

    if (m_negativeMode) {
        qDebug() << "[CTRL][loadDicom][FX] re-applying negative mode";
        applyNegative();
    }

    scheduleOrDoDicomMetadataRead(path);
    qDebug() << "[CTRL][loadDicom] EXIT ok";
    return true;
}

static inline QString qclean(const std::string& s) {
    QString q = QString::fromUtf8(s.c_str());
    if (q.contains(QChar::ReplacementCharacter)) q = QString::fromLatin1(s.c_str());
    return q.trimmed();
}

QStringList AppController::formatDicomMeta(const io::DicomMeta& m) const
{
    auto nz = [](const QString& v, const QString& fallback = QString("—")){
        return v.isEmpty() ? fallback : v;
    };

    const QString manuf  = qclean(m.manufacturer);
    const QString model  = qclean(m.modelName);
    const QString sdate  = qclean(m.studyDate);
    const QString stime  = qclean(m.studyTime);
    const QString b0     = qclean(m.B0T);
    const QString tr     = qclean(m.tr_ms);
    const QString te     = qclean(m.te_ms);
    const QString ti     = qclean(m.ti_ms);
    const QString pname  = qclean(m.patientName);
    const QString pid    = qclean(m.patientID);

    QStringList L;
    L << "=== Acquisition ===";
    if (!manuf.isEmpty() || !model.isEmpty())
        L << QString("System: %1 — %2").arg(nz(manuf)).arg(nz(model));
    if (!sdate.isEmpty() || !stime.isEmpty())
        L << QString("Study Date/Time: %1 %2").arg(nz(sdate, "<unknown>")).arg(nz(stime, "")).trimmed();
    if (!b0.isEmpty())
        L << QString("B0: %1 T").arg(b0);
    if (!tr.isEmpty() || !te.isEmpty() || !ti.isEmpty())
        L << QString("Sequence (ms): TR=%1  TE=%2  TI=%3")
                 .arg(nz(tr)).arg(nz(te)).arg(nz(ti));

    if (!pname.isEmpty() || !pid.isEmpty()) {
        L << "";
        L << "=== Patient ===";
        if (!pname.isEmpty()) L << QString("Name: %1").arg(pname);
        if (!pid.isEmpty())   L << QString("ID: %1").arg(pid);
    }
    return L;
}

cv::Mat AppController::convert16To8(const cv::Mat& f16)
{
    if (f16.empty()) { qWarning() << "[CTRL][convert16To8][WRN] empty input"; return {}; }
    if (f16.depth() != CV_16U) { qWarning() << "[CTRL][convert16To8][WRN] not CV_16U, type=" << f16.type(); }

    double mn = 0.0, mx = 0.0;
    cv::minMaxLoc(f16, &mn, &mx);
    if (mx <= mn) mx = mn + 1.0;

    cv::Mat f32, u8;
    f16.convertTo(f32, CV_32F);
    f32 = (f32 - float(mn)) / float(mx - mn);
    f32.convertTo(u8, CV_8U, 255.0);

    qDebug() << "[CTRL][convert16To8][DBG] min=" << mn << " max=" << mx
             << " out size=" << u8.cols << "x" << u8.rows << " type=" << u8.type();
    return u8;
}

bool AppController::decodeDicomToFrames8(const std::string& path,
                                         std::vector<cv::Mat>& outFrames8,
                                         std::string& why)
{
    qDebug() << "[CTRL][decodeDicomToFrames8] ENTER";
    outFrames8.clear();

    qDebug() << "[CTRL][decodeDicomToFrames8] calling io::read_dicom_frames_u16…";
    std::vector<cv::Mat> frames16;
    bool ok16 = io::read_dicom_frames_u16(path, frames16, &why);

    if (ok16 && !frames16.empty()) {
        qDebug() << "[CTRL][decodeDicomToFrames8] got" << int(frames16.size())
        << "frame(s) 16-bit; converting each to 8-bit…";

        outFrames8.reserve(frames16.size());
        for (size_t i = 0; i < frames16.size(); ++i) {
            const cv::Mat& f16 = frames16[i];
            if (f16.empty()) {
                qWarning() << "[CTRL][decodeDicomToFrames8][WRN] empty 16-bit frame at idx" << int(i);
                continue;
            }
            cv::Mat u8 = convert16To8(f16);
            outFrames8.emplace_back(u8.clone());
            if (i < 3) {
                qDebug() << "[CTRL][decodeDicomToFrames8][DBG] idx=" << int(i)
                << " sz=" << u8.cols << "x" << u8.rows << " type=" << u8.type();
            }
        }
        qDebug() << "[CTRL][decodeDicomToFrames8] 16->8 done; outFrames8=" << int(outFrames8.size());
        qDebug() << "[CTRL][decodeDicomToFrames8] EXIT ok";
        return !outFrames8.empty();
    }

    qWarning() << "[CTRL][decodeDicomToFrames8][WRN] 16-bit path failed:"
               << QString::fromStdString(why) << " -> trying 8-bit fallback";
    why.clear();

    std::vector<cv::Mat> frames8_raw;
    if (!io::read_dicom_frames_gray8(path, frames8_raw, &why) || frames8_raw.empty()) {
        qCritical() << "[CTRL][decodeDicomToFrames8][ERR] All decode attempts failed:"
                    << QString::fromStdString(why);
        qDebug() << "[CTRL][decodeDicomToFrames8] EXIT fail";
        return false;
    }

    qDebug() << "[CTRL][decodeDicomToFrames8] decoded" << int(frames8_raw.size()) << "frame(s) as 8-bit";
    outFrames8.reserve(frames8_raw.size());
    for (size_t i = 0; i < frames8_raw.size(); ++i) {
        const cv::Mat& f = frames8_raw[i];
        if (f.empty()) continue;

        if (f.type() == CV_8UC1) {
            outFrames8.emplace_back(f.clone());
        } else {
            cv::Mat g;
            if (f.channels() == 3) cv::cvtColor(f, g, cv::COLOR_BGR2GRAY);
            else                    f.convertTo(g, CV_8U);
            outFrames8.emplace_back(g.clone());
        }
    }

    qDebug() << "[CTRL][decodeDicomToFrames8] EXIT ok (fallback)";
    return !outFrames8.empty();
}

void AppController::adoptFrames8ToState(const std::vector<cv::Mat>& frames8)
{
    qDebug() << "[CTRL][adoptFrames8ToState] ENTER frames=" << int(frames8.size());
    if (frames8.empty()) { qWarning() << "[CTRL][adoptFrames8ToState][WRN] empty frames"; return; }

    m_slices8_base = frames8;
    m_slices8      = m_slices8_base;
    m_currentSlice = 0;
    m_display8     = m_slices8.front().clone();
    m_lastImg8     = m_display8.clone();

    qDebug() << "[CTRL][adoptFrames8ToState] adopting stack: S="
             << int(m_slices8.size())
             << " W=" << m_display8.cols << " H=" << m_display8.rows;

    if (m_view) {
        m_view->enableSliceSlider(int(m_slices8.size()));
        m_view->setSliceIndex(0);
        m_view->setImage(m_display8);
    } else {
        qWarning() << "[CTRL][adoptFrames8ToState][WRN] m_view is null; cannot display";
    }

    qDebug() << "[CTRL][adoptFrames8ToState] EXIT";
}

void AppController::scheduleOrDoDicomMetadataRead(const std::string& path)
{
    qDebug() << "[CTRL][scheduleOrDoDicomMetadataRead] deferred";

    auto task = [ctrl=this, path]() noexcept {
        try {
            qDebug() << "[CTRL][META] ENTER";
            std::string why2;
            io::DicomMeta meta{};
            const bool ok = io::read_dicom_basic_meta(path, meta, &why2);
            if (!ok) {
                qWarning() << "[CTRL][META][WRN]" << QString::fromStdString(why2);
                qDebug() << "[CTRL][META] EXIT (fail)";
                return;
            }

            QMetaObject::invokeMethod(
                qApp,
                [ctrl, meta]() {
                    if (!ctrl || !ctrl->m_view) {
                        qWarning() << "[CTRL][META] view/controller gone; skipping UI update";
                        return;
                    }
                    qDebug() << "[CTRL][META] ok; updating view";

                    QStringList lines = ctrl->m_meta;
                    auto add = [&](const QString& s) {
                        const QString t = s.trimmed();
                        if (!t.isEmpty()) lines << t;
                    };

                    {
                        QStringList parts;
                        if (!meta.manufacturer.empty())
                            parts << QString::fromStdString(meta.manufacturer);
                        if (!meta.modelName.empty())
                            parts << QString::fromStdString(meta.modelName);

                        if (!parts.isEmpty()) {
                            QString dev = QString("Device: %1").arg(parts.join(" "));
                            if (!meta.B0T.empty())
                                dev += QString(" (B0: %1 T)").arg(QString::fromStdString(meta.B0T));
                            add(dev);
                        }
                    }

                    if (!meta.patientName.empty() || !meta.patientID.empty()) {
                        const QString name = QString::fromStdString(meta.patientName);
                        const QString pid  = QString::fromStdString(meta.patientID);
                        add(pid.isEmpty() ? QString("Patient: %1").arg(name)
                                          : QString("Patient: %1 (%2)").arg(name, pid));
                    }

                    if (!meta.studyDate.empty())
                        add(QString("Study Date: %1").arg(QString::fromStdString(meta.studyDate)));
                    if (!meta.studyTime.empty())
                        add(QString("Study Time: %1").arg(QString::fromStdString(meta.studyTime)));

                    if (!meta.tr_ms.empty())
                        add(QString("TR (ms): %1").arg(QString::fromStdString(meta.tr_ms)));
                    if (!meta.te_ms.empty())
                        add(QString("TE (ms): %1").arg(QString::fromStdString(meta.te_ms)));
                    if (!meta.ti_ms.empty())
                        add(QString("TI (ms): %1").arg(QString::fromStdString(meta.ti_ms)));

                    ctrl->m_meta = lines;
                    ctrl->m_view->setMetadata(ctrl->m_meta);
                },
                Qt::QueuedConnection
                );

            qDebug() << "[CTRL][META] EXIT ok";
        } catch (const std::exception& ex) {
            qCritical() << "[CTRL][META][EXC]" << ex.what();
        } catch (...) {
            qCritical() << "[CTRL][META][EXC] unknown";
        }
    };

    QTimer::singleShot(0,
                       m_view ? static_cast<QObject*>(m_view) : static_cast<QObject*>(qApp),
                       std::move(task));
}
