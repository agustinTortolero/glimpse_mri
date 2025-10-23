#include "mainwindow.hpp"

#include <QAction>
#include <QApplication>
#include <QContextMenuEvent>
#include <QDateTime>
#include <QDir>
#include <QDockWidget>
#include <QDragEnterEvent>
#include <QDragMoveEvent>
#include <QDropEvent>
#include <QEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QImage>
#include <QLabel>
#include <QMenu>
#include <QMimeData>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QResizeEvent>
#include <QScreen>
#include <QSignalBlocker>
#include <QSlider>
#include <QStandardPaths>
#include <QStatusBar>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWheelEvent>
#include <QDebug>

#include <QCoreApplication>
#include <QEventLoop>
#include <QDialog>
#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QDialogButtonBox>
#include <QTextBrowser>
#include <QFont>
#include <QMessageBox>

#include <memory>
#include <iostream>
#include <algorithm>
#include <chrono>

#include <opencv2/imgproc.hpp>
#include <cmath> // std::log10, std::log1p, std::sqrt, std::asinh

namespace {
// Transform raw bin counts into display “heights”, per the chosen scale
static void transformCounts(const cv::Mat& histIn, cv::Mat& histOut,
                            MainWindow::HistScale scale,
                            double eps = 1.0,     // for log
                            double asinhK = 1e-3) // for asinh
{
    histOut.create(histIn.size(), CV_32F);
    for (int i = 0; i < histIn.rows; ++i) {
        float c = histIn.at<float>(i);
        float v = 0.0f;
        switch (scale) {
        case MainWindow::HistScale::Linear: v = c; break;
        case MainWindow::HistScale::Log10:  v = static_cast<float>(std::log10(eps + c)); break;
        case MainWindow::HistScale::Sqrt:   v = std::sqrt(c); break;
        case MainWindow::HistScale::Asinh:  v = static_cast<float>(std::asinh(asinhK * c)); break;
        }
        histOut.at<float>(i) = v;
    }
}
}



using namespace std::chrono_literals;

// ===== Constructor (slider + wheel + DnD) =====
MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent)
{
    std::cerr << "[DBG][View][Ctor] start\n";
    setAcceptDrops(true);                 // enable window-level drag&drop
    buildUi();
    std::cerr << "[DBG][View][Ctor] end\n";
}
MainWindow::~MainWindow()
{
    qDebug() << "[View][dtor] MainWindow";
}

void MainWindow::buildUi()
{
    qDebug() << "[View][buildUi]";
    setCentralWidget(createCentralArea());

    // Create docks (hidden by default; we will show them on first real image)
    createMetadataDock();
    createHistogramDock();

    // Do NOT show docks here; requirement: when no image, no docks visible.
    // Initial window size:
    setInitialSize();

    showDragHint();  // optional startup UX
}


void MainWindow::createHistogramDock()
{
    qDebug() << "[View][createHistogramDock]";

    auto* dock = new QDockWidget(tr("Histogram (grayscale)"), this);
    dock->setObjectName(QStringLiteral("HistDock"));
    dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    dock->setFeatures(QDockWidget::DockWidgetClosable
                      | QDockWidget::DockWidgetMovable
                      | QDockWidget::DockWidgetFloatable);

    auto* label = new QLabel(dock);
    label->setAlignment(Qt::AlignCenter);
    label->setText("No image");
    label->setMinimumSize(100, 80);
    label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    label->installEventFilter(this);  // redraw histogram when resized

    dock->setWidget(label);
    addDockWidget(Qt::RightDockWidgetArea, dock);

    // Hidden by default (no image => no docks visible)
    dock->setFloating(false);
    dock->hide();

    m_histDock  = dock;
    m_histLabel = label;

    qDebug() << "[View][Hist] Dock created (hidden by default)";
}

void MainWindow::setImageCV8U(const cv::Mat& m)
{
    qDebug() << "[View][setImageCV8U] ENTER empty=" << m.empty()
    << " type=" << m.type()
    << " size=" << m.cols << "x" << m.rows;

    if (m.empty()) {
        qWarning() << "[View][setImageCV8U] empty matrix";
        return;
    }

    // Route through the existing cv::Mat pipeline:
    // - converts to 8U mono if needed
    // - stores to m_img8/m_hasImage
    // - updates metadata
    // - repaints image
    // - updates histogram
    setImage(m);

    qDebug() << "[View][setImageCV8U] EXIT";
}


void MainWindow::updateHistogramDock(const cv::Mat& u8)
{
    if (!m_histLabel || !m_histDock) {
        qWarning() << "[View][Hist] widgets missing";
        return;
    }

    // Pick the best available image: prefer argument; else fall back to last stored slice.
    const bool argEmpty   = u8.empty();
    const bool haveStored = m_hasImage && !m_img8.empty();
    const cv::Mat& srcRef = (!argEmpty ? u8 : (haveStored ? m_img8 : u8));
    const bool haveImage  = !srcRef.empty();

    qDebug() << "[View][Hist] update ENTER"
             << "argEmpty=" << argEmpty
             << "haveStored=" << haveStored
             << "haveImage=" << haveImage
             << "neg=" << m_negativeMode;

    // One-time layout init on the first real image only. Never auto re-show after.
    if (haveImage && m_metaDock && m_histDock) {
        const bool firstInit = !m_metaDock->property("sizedOnce").toBool();
        if (firstInit) {
            m_metaDock->show();
            m_histDock->show();

            if (!m_metaDock->property("stackedOnce").toBool()) {
                splitDockWidget(m_metaDock, m_histDock, Qt::Vertical); // meta above, hist below
                m_metaDock->setProperty("stackedOnce", true);
                qDebug() << "[View][Dock] splitDockWidget(meta, hist, Vertical)";
            }

            // Right dock area = 1/2 of window width (once)
            const int halfW = qMax(120, width() / 2);
            QList<QDockWidget*> docksW { m_metaDock }; // any dock in right column works
            QList<int>          sizesW { halfW };
            resizeDocks(docksW, sizesW, Qt::Horizontal);
            m_metaDock->setProperty("sizedOnce", true);
            qDebug() << "[View][Dock] resizeDocks Horizontal -> halfW=" << halfW << " winW=" << width();

            // Split the two right docks' heights 50/50 (once)
            const int halfH = qMax(60, height() / 2);
            QList<QDockWidget*> docksH { m_metaDock, m_histDock };
            QList<int>          sizesH { halfH, halfH };
            resizeDocks(docksH, sizesH, Qt::Vertical);
            m_metaDock->setProperty("vSizedOnce", true);
            qDebug() << "[View][Dock] resizeDocks Vertical -> halfH=" << halfH << " winH=" << height();
        }
    }

    // Canvas size from the label so it scales with the dock.
    const QSize area = m_histLabel->contentsRect().size();
    const int   W    = qMax(100, area.width());
    const int   H    = qMax(80,  area.height());
    qDebug() << "[View][Hist] canvas" << W << "x" << H;

    // Colors (BGR)
    const cv::Scalar bg   = m_negativeMode ? cv::Scalar(0,   0,   0)   : cv::Scalar(255, 255, 255);
    const cv::Scalar bar  =                           /* BLUE */          cv::Scalar(255, 0,   0);
    const cv::Scalar axis = m_negativeMode ? cv::Scalar(90,  90,  90)  : cv::Scalar(180, 180, 180);

    cv::Mat canvas(H, W, CV_8UC3, bg);
    auto drawBaseline = [&](){
        cv::line(canvas, cv::Point(0, H - 1), cv::Point(W - 1, H - 1), axis, 1, cv::LINE_8);
    };

    // If we truly have no image: hide both docks and prepare a clean placeholder.
    if (!haveImage) {
        if (m_metaDock && m_metaDock->isVisible()) m_metaDock->hide();
        if (m_histDock && m_histDock->isVisible()) m_histDock->hide();

        drawBaseline();
        QImage qi(canvas.data, canvas.cols, canvas.rows, int(canvas.step), QImage::Format_BGR888);
        m_histLabel->setPixmap(QPixmap::fromImage(qi.copy()));
        m_histLabel->setText("No image");
        qDebug() << "[View][Hist] no image -> docks hidden + placeholder";
        return;
    }

    // Ensure 8-bit mono input
    cv::Mat src8;
    if (srcRef.type() == CV_8UC1) src8 = srcRef;
    else                          srcRef.convertTo(src8, CV_8U);

    // Raw histogram (256 bins)
    constexpr int histSize = 256;
    float range[2] = {0.f, 256.f};
    const float* ranges[] = { range };
    int channels[] = { 0 };
    cv::Mat hist;
    cv::calcHist(&src8, 1, channels, cv::Mat(), hist, 1, &histSize, ranges, /*uniform*/true, /*accum*/false);

    // ALWAYS ignore background band:
    // normal -> bins 0..4; negative -> bins 251..255
    {
        const int start = m_negativeMode ? qMax(0, hist.rows - 5) : 0;
        const int end   = m_negativeMode ? hist.rows - 1          : qMin(hist.rows - 1, 4);
        double suppressed = 0.0;
        for (int i = start; i <= end; ++i) { suppressed += hist.at<float>(i); hist.at<float>(i) = 0.f; }
        qDebug() << "[View][Hist] background bins ignored:" << start << "..." << end << " suppressed=" << suppressed;
    }

    // Linear scale (no transform)
    double minv = 0.0, maxv = 0.0;
    cv::minMaxLoc(hist, &minv, &maxv);

    // Layout & draw
    const int topMargin   = 12;
    const int leftMargin  = 6;
    const int rightMargin = 6;
    const int usableW     = qMax(1, W - leftMargin - rightMargin);
    const int baseY       = H - 1;

    drawBaseline();

    const double hScale = (maxv > 0.0) ? (double(H - topMargin - 1) / maxv) : 0.0;
    const double binWf  = double(usableW) / double(histSize);

    for (int i = 0; i < histSize; ++i) {
        const int x0 = leftMargin + qBound(0, int(std::floor(i * binWf)),  qMax(0, W - 1));
        int       x1 = leftMargin + qBound(0, int(std::floor((i + 1) * binWf)) - 1, qMax(0, W - 1));
        if (x1 < x0) x1 = x0;

        int h = int(std::round(hist.at<float>(i) * hScale));
        h = qBound(0, h, qMax(0, H - topMargin));

        cv::rectangle(canvas, {x0, baseY}, {x1, baseY - h}, bar, cv::FILLED, cv::LINE_8);
    }

    QImage qi(canvas.data, canvas.cols, canvas.rows, int(canvas.step), QImage::Format_BGR888);
    m_histLabel->setPixmap(QPixmap::fromImage(qi.copy()));
    m_histLabel->setToolTip(QString("Histogram: 256 bins | bg ignored: %1 | size=%2x%3 | canvas=%4x%5")
                                .arg(m_negativeMode ? "251..255" : "0..4")
                                .arg(src8.cols).arg(src8.rows)
                                .arg(W).arg(H));

    qDebug() << "[View][Hist] update EXIT (linear, bg-ignored)";
}






QWidget* MainWindow::createCentralArea()
{
    qDebug() << "[View][createCentralArea]";
    auto* central = new QWidget(this);
    auto* vlay    = new QVBoxLayout(central);
    vlay->setContentsMargins(8, 8, 8, 8);
    vlay->setSpacing(6);

    // --- image area ---
    m_label = new QLabel(central);
    m_label->setAlignment(Qt::AlignCenter);
    m_label->setText("No image");
    m_label->installEventFilter(this);     // wheel navigation
    vlay->addWidget(m_label, /*stretch*/1);

    // --- slice controls row (slider only) ---
    auto* row  = new QWidget(central);
    auto* hlay = new QHBoxLayout(row);
    hlay->setContentsMargins(0,0,0,0);
    hlay->setSpacing(8);

    m_sliceSlider = new QSlider(Qt::Horizontal, row);
    m_sliceSlider->setEnabled(false);        // disabled until controller enables
    m_sliceSlider->setRange(0, 0);           // 0..0 until we know the stack size
    m_sliceSlider->setValue(0);
    m_sliceSlider->setVisible(false);        // hidden when single/no slice

    hlay->addWidget(m_sliceSlider, /*stretch*/1);
    row->setLayout(hlay);

    vlay->addWidget(row, /*stretch*/0);

    // connect slider -> slot -> signal to controller
    connect(m_sliceSlider, &QSlider::valueChanged,
            this, &MainWindow::onSliderValueChanged);

    return central;
}

void MainWindow::createMetadataDock()
{
    qDebug() << "[View][createMetadataDock]";

    auto* dock = new QDockWidget(tr("Image details"), this);
    dock->setObjectName(QStringLiteral("MetaDock"));
    dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    dock->setFeatures(QDockWidget::DockWidgetClosable
                      | QDockWidget::DockWidgetMovable
                      | QDockWidget::DockWidgetFloatable);

    auto* txt = new QPlainTextEdit(dock);
    txt->setReadOnly(true);
    dock->setWidget(txt);

    addDockWidget(Qt::RightDockWidgetArea, dock);

    // Hidden by default: "no image -> no docks visible"
    dock->setFloating(false);
    dock->hide();

    m_metaDock = dock;
    m_metaText = txt;

    // Guards so we only stack/size once when the first real image arrives
    m_metaDock->setProperty("stackedOnce", false);
    m_metaDock->setProperty("sizedOnce",   false);  // width (½ window) guard
    m_metaDock->setProperty("vSizedOnce",  false);  // NEW: height split (50/50) guard

    qDebug() << "[View][Meta] Dock created (hidden by default)";
}




void MainWindow::setInitialSize()
{
    resize(900, 571);
    qDebug() << "[View] Initial size set to" << width() << "x" << height();
}

// ==============================
// Events / Filters
// ==============================
bool MainWindow::eventFilter(QObject* obj, QEvent* ev)
{
    // Wheel over image (existing behavior)
    if (obj == m_label && ev->type() == QEvent::Wheel) {
        auto* wev = static_cast<QWheelEvent*>(ev);
        const int delta = wev->angleDelta().y();
        if (delta != 0 && m_sliceSlider && m_sliceSlider->isEnabled()) {
            const int step = (delta > 0 ? -1 : +1);
            int v = m_sliceSlider->value() + step;
            v = qBound(m_sliceSlider->minimum(), v, m_sliceSlider->maximum());
            qDebug() << "[View][Wheel] step=" << step << " -> slider" << m_sliceSlider->value() << "->" << v;
            m_sliceSlider->setValue(v);
            return true;
        }
        return false;
    }

    if (obj == m_histLabel && ev->type() == QEvent::Resize) {
        qDebug() << "[View][Hist] label resized -> redraw";
        if (m_histDock && m_histDock->isVisible()) {
            if (m_hasImage && !m_img8.empty()) updateHistogramDock(m_img8);
            else {
                cv::Mat empty;
                updateHistogramDock(empty);
            }
        }
        return false;
    }

    return QMainWindow::eventFilter(obj, ev);
}


// ==============================
// Metadata helpers
// ==============================
void MainWindow::setMetadata(const QStringList& lines)
{
    if (!m_metaText) return;
    m_metaText->setPlainText(lines.join('\n'));
    qDebug() << "[View][Meta] setMetadata with" << lines.size() << "line(s)";
}

void MainWindow::appendMetadataLine(const QString& line)
{
    if (!m_metaText) return;
    m_metaText->appendPlainText(line);
    qDebug().noquote() << "[View][Meta] append:" << line;
}

void MainWindow::beginNewImageCycle()
{
    qDebug() << "[View] beginNewImageCycle(): clear image + slider state";
    m_hasImage = false;
    m_img8.release();

    if (m_label) {
        m_label->clear();
        showDragHint();
    }
    enableSliceSlider(0);  // disables & hides slider

    // Requirement: when no image, no docks visible
    if (m_metaDock && m_metaDock->isVisible()) m_metaDock->hide();
    if (m_histDock && m_histDock->isVisible()) m_histDock->hide();

    // Clear contents so we start fresh next time
    if (m_metaText)  m_metaText->setPlainText(QString());
    if (m_histLabel) m_histLabel->setText("No image");

    qDebug() << "[View] Docks hidden (no image)";
}



// ---- setImage refactored into small helpers ----
bool MainWindow::validateImageInput(const cv::Mat& img) const
{
    const bool ok = !img.empty();
    if (!ok) qWarning() << "[View] setImage: empty; ignoring";
    return ok;
}

cv::Mat MainWindow::to8uMono(const cv::Mat& src) const
{
    if (src.type() == CV_8UC1) {
        qDebug() << "[View] setImage: already 8U mono";
        return src.clone();
    }
    qDebug() << "[View] setImage: converting to 8U mono";
    cv::Mat dst; src.convertTo(dst, CV_8U); return dst;
}

void MainWindow::logMatStats(const cv::Mat& m) const
{
    double minv = 0.0, maxv = 0.0;
    cv::minMaxLoc(m, &minv, &maxv);
    qDebug() << "[View] setImage: m_img8 size=" << m.cols << "x" << m.rows
             << " min=" << minv << " max=" << maxv
             << " type=" << m.type();
}

void MainWindow::storeImage(const cv::Mat& m)
{
    m_img8     = m.clone();
    m_hasImage = true;
}

void MainWindow::updateMetadataForImage(const cv::Mat& m)
{
    if (!m_metaText) return;
    double minv = 0.0, maxv = 0.0;
    cv::minMaxLoc(m, &minv, &maxv);
    const QString ts = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss");
    appendMetadataLine(QString("[View] %1 | Image %2x%3 | min=%4 max=%5")
                           .arg(ts).arg(m.cols).arg(m.rows)
                           .arg(minv, 0, 'f', 3).arg(maxv, 0, 'f', 3));
}

void MainWindow::repaintOnce()
{
    qDebug() << "[View] setImage: repaintOnce -> refreshPixmap()";
    refreshPixmap();
}

void MainWindow::setImage(const cv::Mat& img8u)
{
    qDebug() << "[View] setImage ENTER"
             << " empty=" << img8u.empty()
             << " type="  << img8u.type()
             << " dims="  << img8u.cols << "x" << img8u.rows;

    if (!validateImageInput(img8u)) return;

    clearDragHint();
    cv::Mat m = to8uMono(img8u);      // ensures 8-bit grayscale
    logMatStats(m);
    storeImage(m);
    updateMetadataForImage(m);
    repaintOnce();

    // NEW: refresh histogram of the current slice
    updateHistogramDock(m);

    qDebug() << "[View] setImage EXIT";
}


// ==============================
// Painting (refreshPixmap refactor)
// ==============================
bool MainWindow::beginRefreshGuard()
{
    if (m_refreshing) {
        qDebug() << "[View] refreshPixmap: already refreshing; skip";
        return false;
    }
    m_refreshing = true;
    return true;
}

void MainWindow::endRefreshGuard()
{
    m_refreshing = false;
}

bool MainWindow::hasDrawableImage() const
{
    const bool ok = (m_hasImage && !m_img8.empty() && m_label);
    if (!ok) qDebug() << "[View] refreshPixmap: no image or label; skipping";
    return ok;
}

bool MainWindow::labelTooSmall() const
{
    if (!m_label) return true;
    const QSize s = m_label->size();
    if (s.width() <= 1 || s.height() <= 1) {
        qDebug() << "[View] refreshPixmap: label too small (" << s.width() << "x" << s.height() << "); skip once";
        return true;
    }
    return false;
}

bool MainWindow::isMultiSliceActive() const
{
    if (!m_sliceSlider) return false;
    const int S = m_sliceSlider->maximum() - m_sliceSlider->minimum() + 1;
    return (m_sliceSlider->isEnabled() && S >= 2);
}

cv::Mat MainWindow::buildDisplayImageWithOverlay()
{
    if (!isMultiSliceActive()) {
        return m_img8;                       // no copy, no overlay
    }
    cv::Mat shown = m_img8.clone();          // keep original pristine
    drawSliceOverlay(shown);
    return shown;                             // copy with overlay
}

QImage MainWindow::toQImageOwned(const cv::Mat& m) const
{
    QImage q(m.data, m.cols, m.rows, static_cast<int>(m.step), QImage::Format_Grayscale8);
    return q.copy(); // own the pixels
}

QImage MainWindow::scaleForLabel(const QImage& qi) const
{
    const QSize lbl = m_label ? m_label->size() : QSize(0,0);
    const Qt::AspectRatioMode ar = Qt::KeepAspectRatio;
    const Qt::TransformationMode mode = Qt::FastTransformation;
    return qi.scaled(lbl, ar, mode);
}

void MainWindow::setPixmapAndLog(const QImage& scaled)
{
    QPixmap pm = QPixmap::fromImage(scaled);
    if (m_label) {
        m_label->setPixmap(pm);
        m_label->update();
    }
    const QSize lbl = m_label ? m_label->size() : QSize(0,0);
    qDebug() << "[View] refreshPixmap EXIT  src="
             << m_img8.cols << "x" << m_img8.rows
             << " label=" << lbl.width() << "x" << lbl.height()
             << " pixmap=" << pm.width() << "x" << pm.height();
}

void MainWindow::refreshPixmap()
{
    if (!beginRefreshGuard()) return;

    if (!hasDrawableImage()) { endRefreshGuard(); return; }
    if (labelTooSmall())     { endRefreshGuard(); return; }

    cv::Mat disp = buildDisplayImageWithOverlay();
    QImage  qi   = toQImageOwned(disp);
    QImage  sc   = scaleForLabel(qi);
    setPixmapAndLog(sc);

    endRefreshGuard();
}

void MainWindow::resizeEvent(QResizeEvent* ev)
{
    QMainWindow::resizeEvent(ev);
    qDebug() << "[View] resizeEvent -> refreshPixmap()";
    refreshPixmap();
}

// ==============================
// Busy UI
// ==============================
void MainWindow::beginBusy(const QString& message)
{
    ++m_busyNesting;
    QApplication::setOverrideCursor(Qt::WaitCursor);
    if (statusBar()) statusBar()->showMessage(message);
    qDebug() << "[View] beginBusy:" << message << " nesting=" << m_busyNesting;
}

void MainWindow::endBusy()
{
    if (m_busyNesting > 0) --m_busyNesting;
    if (m_busyNesting == 0) {
        QApplication::restoreOverrideCursor();
        if (statusBar()) statusBar()->clearMessage();
        qDebug() << "[View] endBusy (restored cursor)";
    } else {
        qDebug() << "[View] endBusy (nested level =" << m_busyNesting << ")";
    }
}

// ==============================
// Slice slider API (controller-facing)
// ==============================
void MainWindow::enableSliceSlider(int nSlices)
{
    if (!m_sliceSlider) return;

    QSignalBlocker block(m_sliceSlider);

    if (nSlices >= 2) {
        m_sliceSlider->setEnabled(true);
        m_sliceSlider->setVisible(true);
        m_sliceSlider->setRange(0, nSlices - 1);
        m_sliceSlider->setValue(0);
        qDebug() << "[View][Slider] enabled with" << nSlices << "slices";
        if (statusBar()) statusBar()->showMessage(QString("Slices: %1").arg(nSlices), 1500);
    } else {
        m_sliceSlider->setEnabled(false);
        m_sliceSlider->setVisible(false);
        m_sliceSlider->setRange(0, 0);
        m_sliceSlider->setValue(0);
        qDebug() << "[View][Slider] disabled (single/no slice)";
    }

    // when the slider state changes, repaint once so overlay presence matches
    refreshPixmap();

    qDebug() << "[View][Slider] state -> enabled=" << m_sliceSlider->isEnabled()
             << " visible=" << m_sliceSlider->isVisible()
             << " value=" << m_sliceSlider->value();
}

void MainWindow::setSliceIndex(int idx)
{
    if (!m_sliceSlider) return;
    QSignalBlocker block(m_sliceSlider); // avoid emitting sliceChanged
    idx = std::clamp(idx, m_sliceSlider->minimum(), m_sliceSlider->maximum());
    m_sliceSlider->setValue(idx);
    qDebug() << "[View][Slider] setSliceIndex ->" << idx << "(signals blocked)";
    // ensure overlay text updates immediately
    refreshPixmap();
}

void MainWindow::onSliderValueChanged(int v)
{
    qDebug() << "[View][Slider] valueChanged -> emit sliceChanged(" << v << ")";
    emit sliceChanged(v);
    // overlay index on image should update as the controller pushes the new slice;
    // still, a local repaint is safe if image content didn't change
    refreshPixmap();
}

// ==============================
// Negative mode (controller sync)
// ==============================
void MainWindow::onNegativeModeChanged(bool on)
{
    m_negativeMode = on;
    qDebug() << "[View] onNegativeModeChanged ->" << on;

    // If the histogram dock is visible, redraw it to adapt background color
    if (m_histDock && m_histDock->isVisible()) {
        if (m_hasImage && !m_img8.empty()) updateHistogramDock(m_img8);
        else {
            cv::Mat empty;
            updateHistogramDock(empty);
        }
    }
}


// ==============================
// Drag & Drop (fileDropped)
// ==============================
bool MainWindow::isAcceptableUrl(const QUrl& url) const
{
    if (!url.isLocalFile()) return false;
    return isAcceptablePath(url.toLocalFile());
}

bool MainWindow::isAcceptablePath(const QString& path) const
{
    QFileInfo fi(path);
    if (!fi.exists()) return false;

    if (fi.isDir()) {
        // Allow directories (DICOM folders)
        qDebug() << "[DnD][Check] dir accepted:" << path;
        return true;
    }

    const QString ext = fi.suffix().toLower();
    const bool ok = m_okExts.contains(ext);
    qDebug() << "[DnD][Check] file ext=" << ext << " ok=" << ok << " path=" << path;
    return ok;
}

void MainWindow::dragEnterEvent(QDragEnterEvent* ev)
{
    const QMimeData* md = ev->mimeData();
    if (!md->hasUrls()) {
        qDebug() << "[DnD] dragEnter: no URLs -> ignore";
        ev->ignore();
        return;
    }
    // Accept if at least one acceptable item
    for (const QUrl& u : md->urls()) {
        if (isAcceptableUrl(u)) {
            qDebug() << "[DnD] dragEnter: accepting";
            ev->acceptProposedAction();
            return;
        }
    }
    qDebug() << "[DnD] dragEnter: no acceptable items";
    ev->ignore();
}

void MainWindow::dragMoveEvent(QDragMoveEvent* ev)
{
    // Same policy as dragEnter
    const QMimeData* md = ev->mimeData();
    if (md && md->hasUrls()) {
        for (const QUrl& u : md->urls()) {
            if (isAcceptableUrl(u)) {
                ev->acceptProposedAction();
                return;
            }
        }
    }
    ev->ignore();
}

void MainWindow::dropEvent(QDropEvent* ev)
{
    const QMimeData* md = ev->mimeData();
    if (!md || !md->hasUrls()) {
        qDebug() << "[DnD] dropEvent: no URLs";
        ev->ignore();
        return;
    }

    // Strategy: take the first acceptable URL
    for (const QUrl& u : md->urls()) {
        if (!isAcceptableUrl(u)) continue;
        const QString path = u.toLocalFile();
        qDebug() << "[DnD] dropEvent: picked path =" << path;
        if (statusBar()) statusBar()->showMessage(QString("Dropped: %1").arg(path), 2000);
        emit fileDropped(path);
        ev->acceptProposedAction();
        return;
    }

    qDebug() << "[DnD] dropEvent: nothing acceptable";
    ev->ignore();
}

// ==============================
// Context menu (refactored earlier)
// ==============================
void MainWindow::contextMenuEvent(QContextMenuEvent* ev)
{
    const bool hasImg   = hasImageForMenu();
    const bool hasMulti = hasMultiSlicesForMenu();

    qDebug() << "[UI][Menu] contextMenuEvent at" << ev->globalPos()
             << " hasImg=" << hasImg
             << " hasMulti=" << hasMulti;

    CtxMenuActions acts;
    std::unique_ptr<QMenu> menu(buildContextMenu(hasImg, acts));
    if (!menu) {
        qWarning() << "[UI][Menu][WRN] buildContextMenu returned null";
        return;
    }

    QAction* chosen = menu->exec(ev->globalPos());
    if (!chosen) {
        qDebug() << "[UI][Menu] dismissed (no selection)";
        return;
    }

    applyContextSelection(chosen, acts);
}

bool MainWindow::hasImageForMenu() const
{
    const bool ok = (m_hasImage && !m_img8.empty());
    qDebug() << "[UI][Menu] hasImageForMenu =" << ok;
    return ok;
}

bool MainWindow::hasMultiSlicesForMenu() const
{
    if (!m_sliceSlider) return false;
    const int S = m_sliceSlider->maximum() - m_sliceSlider->minimum() + 1;
    const bool ok = (m_sliceSlider->isEnabled() && S >= 2);
    qDebug() << "[UI][Menu] hasMultiSlicesForMenu =" << ok << " S=" << S;
    return ok;
}

QMenu* MainWindow::buildContextMenu(bool hasImg, CtxMenuActions& out)
{
    auto* menu = new QMenu(this);

    if (!hasImg) {
        populateMenuForNoImage(*menu, out);
        return menu;
    }

    const bool hasMulti = hasMultiSlicesForMenu();
    populateMenuForImage(*menu, out, hasMulti, hasImg);
    return menu;
}

void MainWindow::populateMenuForNoImage(QMenu& menu, CtxMenuActions& out)
{
    qDebug() << "[UI][Menu] populateMenuForNoImage";
    out.about = menu.addAction("About");
}

void MainWindow::populateMenuForImage(QMenu& menu,
                                      CtxMenuActions& out,
                                      bool hasMulti,
                                      bool hasImg)
{
    qDebug() << "[UI][Menu] populateMenuForImage hasImg=" << hasImg
             << " hasMulti=" << hasMulti;

    out.saveSlice = menu.addAction("Save slice...");
    out.saveBatch = menu.addAction("Save batch...");
    menu.addSeparator();

    // Negative toggle
    out.negative  = menu.addAction("Negative");
    out.negative->setCheckable(true);
    out.negative->setChecked(m_negativeMode);
    out.negative->setEnabled(hasImg);

    // Histogram toggle (checkable), right under Negative
    QAction* histAct = menu.addAction("Histogram");
    histAct->setObjectName(QStringLiteral("action_histogram"));
    histAct->setCheckable(true);
    const bool histVisible = (m_histDock && m_histDock->isVisible());
    histAct->setChecked(histVisible);
    histAct->setEnabled(hasImg);

    // NEW: Image Info toggle (checkable) -> show/hide metadata dock
    QAction* infoAct = menu.addAction("Image Info");
    infoAct->setObjectName(QStringLiteral("action_image_info"));
    infoAct->setCheckable(true);
    const bool infoVisible = (m_metaDock && m_metaDock->isVisible());
    infoAct->setChecked(infoVisible);
    infoAct->setEnabled(hasImg);

    // No scale submenu, no ignore toggle anymore: histogram is always linear
    // and we always ignore background bins (first 5 / last 5 when negative).

    menu.addSeparator();
    out.startOver = menu.addAction("Start over");

    if (out.saveSlice) out.saveSlice->setEnabled(hasImg);
    if (out.saveBatch) out.saveBatch->setEnabled(hasImg && hasMulti);
}


void MainWindow::applyContextSelection(QAction* chosen, const CtxMenuActions& acts)
{
    if (!chosen) { qDebug() << "[UI][Menu] no action selected"; return; }

    // Histogram toggle
    if (chosen->objectName() == QLatin1String("action_histogram")) {
        qDebug() << "[UI][Menu] Histogram toggled";
        if (!m_histDock) createHistogramDock();
        const bool show = !m_histDock->isVisible();
        if (show) {
            if (m_hasImage && !m_img8.empty()) updateHistogramDock(m_img8);
            m_histDock->show();
            m_histDock->raise();
            qDebug() << "[UI][Hist] dock -> SHOW";
        } else {
            m_histDock->hide();
            qDebug() << "[UI][Hist] dock -> HIDE";
        }
        return;
    }

    // NEW: Image Info toggle -> show/hide metadata dock
    if (chosen->objectName() == QLatin1String("action_image_info")) {
        qDebug() << "[UI][Menu] Image Info toggled";
        if (!m_metaDock) createMetadataDock();
        const bool show = !m_metaDock->isVisible();
        if (show) {
            // Make sure the right area is stacked meta (top) over hist (bottom)
            if (m_histDock && !m_metaDock->property("stackedOnce").toBool()) {
                splitDockWidget(m_metaDock, m_histDock, Qt::Vertical);
                m_metaDock->setProperty("stackedOnce", true);
                qDebug() << "[View][Dock] splitDockWidget(meta, hist, Vertical)";
            }
            m_metaDock->show();
            m_metaDock->raise();
            qDebug() << "[UI][Meta] dock -> SHOW";
        } else {
            m_metaDock->hide();
            qDebug() << "[UI][Meta] dock -> HIDE";
        }
        return;
    }

    // Existing actions
    if (chosen == acts.saveSlice) { qDebug() << "[UI][Menu] Save slice selected"; onSavePNG(); return; }
    if (chosen == acts.saveBatch) { qDebug() << "[UI][Menu] Save batch selected"; onSaveBatch(); return; }
    if (chosen == acts.negative)  { qDebug() << "[UI][Menu] Negative toggled -> emit requestApplyNegative()"; emit requestApplyNegative(); return; }
    if (chosen == acts.startOver) { qDebug() << "[UI][Menu] Start over selected -> emit startOverRequested()"; emit startOverRequested(); return; }
    if (chosen == acts.about)     { qDebug() << "[UI][Menu] About selected -> showAboutDialog()"; showAboutDialog(); return; }

    qDebug() << "[UI][Menu] applyContextSelection: unhandled action";
}



// ==============================
// Save actions
// ==============================
void MainWindow::onSavePNG()
{
    // This slot now handles BOTH PNG and DICOM. We keep the name to avoid header changes.
    if (!m_hasImage || m_img8.empty()) {
        qWarning() << "[UI] onSavePNG(unified): no image to save";
        return;
    }

    // Default directory and base name
    const QString pics = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    const QString baseDir = (pics.isEmpty() ? QDir::homePath() : pics) + "/Glimpse";
    if (!QDir(baseDir).exists()) {
        const bool ok = QDir().mkpath(baseDir);
        qDebug() << "[UI] onSavePNG(unified): mkpath(" << baseDir << ") ->" << (ok ? "OK" : "FAIL");
    }

    QFileDialog dlg(this, tr("Save slice"), baseDir);
    dlg.setAcceptMode(QFileDialog::AcceptSave);
    dlg.setFileMode(QFileDialog::AnyFile);
    // dlg.setOption(QFileDialog::DontUseNativeDialog, true); // enable if dropdown hidden

    // Two explicit filters; user can switch in the combobox
    QStringList filters;
    filters << "PNG (*.png)"
            << "DICOM (*.dcm)";
    dlg.setNameFilters(filters);
    dlg.selectNameFilter("PNG (*.png)");   // default
    dlg.selectFile("image");               // suggest a base file name

    if (!dlg.exec()) {
        qDebug() << "[UI] onSavePNG(unified): user canceled";
        return;
    }

    const QString chosenFilter = dlg.selectedNameFilter();
    QString fn = dlg.selectedFiles().isEmpty() ? QString() : dlg.selectedFiles().front();
    if (fn.isEmpty()) {
        qWarning() << "[UI] onSavePNG(unified): empty filename after exec()";
        return;
    }

    // Decide format (extension overrides filter)
    enum class OutFmt { PNG, DICOM };
    OutFmt fmt = OutFmt::PNG;

    const QString ext = QFileInfo(fn).suffix().toLower();
    if (ext == "png") {
        fmt = OutFmt::PNG;
    } else if (ext == "dcm") {
        fmt = OutFmt::DICOM;
    } else if (chosenFilter.contains("DICOM", Qt::CaseInsensitive)) {
        fmt = OutFmt::DICOM;
    } else {
        fmt = OutFmt::PNG;
    }

    // Append extension if user omitted it
    if (fmt == OutFmt::PNG && !fn.endsWith(".png", Qt::CaseInsensitive)) {
        fn += ".png";
    }
    if (fmt == OutFmt::DICOM && !fn.endsWith(".dcm", Qt::CaseInsensitive)) {
        fn += ".dcm";
    }

    qDebug() << "[UI] onSavePNG(unified) choice:"
             << "file=" << fn
             << " filter=" << chosenFilter
             << " fmt=" << (fmt == OutFmt::PNG ? "PNG" : "DICOM");

    // Emit the existing signals so controller code stays the same
    if (fmt == OutFmt::PNG) {
        qDebug() << "[UI] emit requestSavePNG ->" << fn;
        emit requestSavePNG(fn);
    } else {
        qDebug() << "[UI] emit requestSaveDICOM ->" << fn;
        emit requestSaveDICOM(fn);
    }
}

void MainWindow::onSaveDICOM()
{
    if (!m_hasImage || m_img8.empty()) {
        qWarning() << "[UI] onSaveDICOM: no image to save";
        return;
    }

    QString pics = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    QString suggested = (pics.isEmpty() ? QDir::homePath() : pics) + "/Glimpse/image.dcm";
    qDebug() << "[UI] PicturesLocation =" << pics << "| suggested =" << suggested;

    QString fn = QFileDialog::getSaveFileName(this, "Save as DICOM", suggested, "DICOM (*.dcm)");
    if (fn.isEmpty()) {
        qDebug() << "[UI] onSaveDICOM: user canceled";
        return;
    }
    if (!fn.endsWith(".dcm", Qt::CaseInsensitive)) fn += ".dcm";
    qDebug() << "[UI] requestSaveDICOM ->" << fn;
    emit requestSaveDICOM(fn);
}

void MainWindow::onSaveBatch()
{
    if (!m_hasImage || m_img8.empty() || !m_sliceSlider) {
        qWarning() << "[UI] onSaveBatch: no image/slider";
        return;
    }
    const int S = m_sliceSlider->maximum() - m_sliceSlider->minimum() + 1;
    if (!m_sliceSlider->isEnabled() || S < 2) {
        qWarning() << "[UI] onSaveBatch: only one slice -> disabled";
        return;
    }

    // Where to save and which format
    const QString pics = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    const QString baseDir = (pics.isEmpty() ? QDir::homePath() : pics) + "/Glimpse";
    QDir().mkpath(baseDir);

    QFileDialog dlg(this, tr("Save batch (all slices)"), baseDir);
    dlg.setAcceptMode(QFileDialog::AcceptSave);
    dlg.setFileMode(QFileDialog::AnyFile);
    // dlg.setOption(QFileDialog::DontUseNativeDialog, true); // enable if dropdown hidden

    QStringList filters; filters << "PNG (*.png)" << "DICOM (MR series) (*.dcm)";
    dlg.setNameFilters(filters);
    dlg.selectNameFilter("PNG (*.png)");
    dlg.selectFile("series");

    if (!dlg.exec()) {
        qDebug() << "[UI] onSaveBatch: user canceled";
        return;
    }

    const QString chosenFilter = dlg.selectedNameFilter();
    QString basePath = dlg.selectedFiles().isEmpty() ? QString() : dlg.selectedFiles().front();
    if (basePath.isEmpty()) {
        qWarning() << "[UI] onSaveBatch: empty basePath after exec()";
        return;
    }

    enum class OutFmt { PNG, DICOM_MR_SERIES };
    OutFmt fmt = OutFmt::PNG;
    const QString extTyped = QFileInfo(basePath).suffix().toLower();
    if (extTyped == "png")       fmt = OutFmt::PNG;
    else if (extTyped == "dcm")  fmt = OutFmt::DICOM_MR_SERIES;
    else if (chosenFilter.contains("DICOM", Qt::CaseInsensitive)) fmt = OutFmt::DICOM_MR_SERIES;
    else                        fmt = OutFmt::PNG;

    const QString finalExt = (fmt == OutFmt::PNG) ? ".png" : ".dcm";
    if (!basePath.endsWith(finalExt, Qt::CaseInsensitive)) basePath += finalExt;

    const QFileInfo fi(basePath);
    const QString dir   = fi.absolutePath();
    const QString stem  = fi.completeBaseName();
    const QString ext   = fi.suffix().toLower();

    qDebug() << "[UI] onSaveBatch choice:"
             << "format=" << (fmt == OutFmt::PNG ? "PNG-many" : "DICOM MR series-many")
             << "basePath=" << basePath
             << "S=" << S;

    if (fmt == OutFmt::PNG) {
        // Multi-FILE PNG: base_####.png by stepping the slider
        int pad = 2; if      (S >= 10000) pad = 5;
        else if (S >= 1000 ) pad = 4;
        else if (S >= 100  ) pad = 3;

        int saved = 0;
        const int minIdx = m_sliceSlider->minimum();
        for (int k = 0; k < S; ++k) {
            const int idx = minIdx + k;
            const QString idxStr = QString("%1").arg(k + 1, pad, 10, QChar('0'));
            const QString outPath = dir + "/" + stem + "_" + idxStr + "." + ext;

            { QSignalBlocker block(*m_sliceSlider); m_sliceSlider->setValue(idx); }
            emit sliceChanged(idx);
            QCoreApplication::processEvents(QEventLoop::AllEvents, 8);

            qDebug() << "[UI][Batch-PNG] emit requestSavePNG ->" << outPath;
            emit requestSavePNG(outPath);
            ++saved;
        }
        if (statusBar()) statusBar()->showMessage(
                QString("Saved %1 PNG file(s) to %2").arg(saved).arg(dir), 3000);
        qDebug() << "[UI] onSaveBatch PNG EXIT saved=" << saved;
        return;
    }

    // ============= DICOM MR SERIES (one file per slice) =============
    // Simple geometry dialog (Pixel Spacing, Slice Thickness/Spacing, IOP, IPP0)
    QDialog geo(this);
    geo.setWindowTitle("MR Series Geometry");
    auto* lay = new QFormLayout(&geo);

    auto mkDsb = [&](double val, double min, double max, int dec)->QDoubleSpinBox*{
        auto* s = new QDoubleSpinBox(&geo);
        s->setRange(min, max);
        s->setDecimals(dec);
        s->setSingleStep((max-min)/100.0);
        s->setValue(val);
        s->setMinimumWidth(140);
        return s;
    };

    // Defaults: axial, 1mm spacing, origin at (0,0,0)
    auto* spx = mkDsb(1.0, 0.01, 100.0, 3);
    auto* spy = mkDsb(1.0, 0.01, 100.0, 3);
    auto* sth = mkDsb(1.0, 0.01, 200.0, 3);
    auto* sbs = mkDsb(1.0, 0.00, 200.0, 3);

    auto* r1 = mkDsb(1.0, -1.0, 1.0, 6);
    auto* r2 = mkDsb(0.0, -1.0, 1.0, 6);
    auto* r3 = mkDsb(0.0, -1.0, 1.0, 6);
    auto* c1 = mkDsb(0.0, -1.0, 1.0, 6);
    auto* c2 = mkDsb(1.0, -1.0, 1.0, 6);
    auto* c3 = mkDsb(0.0, -1.0, 1.0, 6);

    auto* x0 = mkDsb(0.0, -10000.0, 10000.0, 3);
    auto* y0 = mkDsb(0.0, -10000.0, 10000.0, 3);
    auto* z0 = mkDsb(0.0, -10000.0, 10000.0, 3);

    lay->addRow(new QLabel("<b>Pixel Spacing (mm)</b>"));
    lay->addRow("dx (row spacing, Y):", spx);
    lay->addRow("dy (col spacing, X):", spy);
    lay->addRow(new QLabel("<b>Slice geometry (mm)</b>"));
    lay->addRow("Slice Thickness:", sth);
    lay->addRow("Spacing Between Slices:", sbs);
    lay->addRow(new QLabel("<b>ImageOrientationPatient (row & col)</b>"));
    lay->addRow("r1 r2 r3:", r1);
    lay->addRow("c1 c2 c3:", c1);
    lay->addRow(new QLabel("<b>ImagePositionPatient (slice #1)</b>"));
    lay->addRow("x0:", x0);
    lay->addRow("y0:", y0);
    lay->addRow("z0:", z0);

    auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &geo);
    lay->addRow(bb);
    QObject::connect(bb, &QDialogButtonBox::accepted, &geo, &QDialog::accept);
    QObject::connect(bb, &QDialogButtonBox::rejected, &geo, &QDialog::reject);

    if (!geo.exec()) {
        qDebug() << "[UI] onSaveBatch: geometry dialog canceled";
        return;
    }

    QVector<double> iop6{ r1->value(), r2->value(), r3->value(),
                         c1->value(), c2->value(), c3->value() };
    QVector<double> ipp0{ x0->value(), y0->value(), z0->value() };

    qDebug() << "[UI][Batch-DICOM-MR] geometry:"
             << "px=" << spx->value() << "py=" << spy->value()
             << "thick=" << sth->value() << "spacing=" << sbs->value()
             << "iop6=" << iop6 << "ipp0=" << ipp0;

    // Emit a single request; the controller will iterate slices and write N files.
    emit requestSaveDICOMSeriesMR(basePath,
                                  spx->value(), spy->value(),
                                  sth->value(), sbs->value(),
                                  iop6, ipp0);

    if (statusBar()) statusBar()->showMessage(
            QString("Writing MR series to %1 (N files)").arg(dir), 2500);
}

// ==============================
// Drag hint helpers
// ==============================
void MainWindow::showDragHint()
{
    if (!m_label) return;
    m_label->setText("Drag DICOM or HDF5\n(.dcm, .ima, .h5, .hdf5)\n—or a DICOM folder—");
    qDebug() << "[DnD][View] drag hint shown";
}

void MainWindow::clearDragHint()
{
    if (!m_label) return;
    if (m_hasImage) {
        // Image will replace pixmap; nothing to draw here.
        qDebug() << "[DnD][View] drag hint cleared (image present)";
    } else {
        m_label->clear();
        qDebug() << "[DnD][View] drag hint cleared (no image yet)";
    }
}

// ==============================
// Overlay for slice index (on-image)
// ==============================
void MainWindow::drawSliceOverlay(cv::Mat& img8)
{
    if (img8.empty()) return;
    if (!m_sliceSlider || !m_sliceSlider->isEnabled()) return;

    const int S = m_sliceSlider->maximum() - m_sliceSlider->minimum() + 1;
    if (S < 2) return; // single image -> no overlay

    const int s = m_sliceSlider->value(); // 0-based

    cv::Mat c3;
    cv::cvtColor(img8, c3, cv::COLOR_GRAY2BGR);

    const std::string txt = "Slice " + std::to_string(s + 1) + "/" + std::to_string(S);

    // layout: size + background box (semi-transparent)
    const double fontScale = 0.6;
    const int    thin      = 1;
    int baseline = 0;
    cv::Size ts = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, fontScale, thin, &baseline);

    cv::Point org(12, 24); // text baseline
    cv::Rect box(org.x - 8, org.y - ts.height - 8, ts.width + 16, ts.height + baseline + 16);
    box &= cv::Rect(0, 0, c3.cols, c3.rows); // clamp to image

    if (box.area() > 0) {
        cv::Mat roi = c3(box);
        cv::Mat overlay = roi.clone();
        cv::rectangle(overlay, cv::Rect(0, 0, roi.cols, roi.rows), cv::Scalar(0, 0, 0), cv::FILLED);
        const double alpha = 0.35; // 35% black
        cv::addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi);
    }

    // solid text: draw black outline first, then white fill
    cv::putText(c3, txt, org, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(c3, txt, org, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    cv::cvtColor(c3, img8, cv::COLOR_BGR2GRAY);

    qDebug() << "[View][Overlay]" << QString::fromStdString(txt)
             << "box=" << box.width << "x" << box.height
             << "alpha=0.35";
}

// ==============================
// About dialog (simple)
// ==============================
void MainWindow::showAboutDialog()
{
    qDebug() << "[About][UI] showAboutDialog ENTER";

    QDialog dlg(this);
    dlg.setWindowTitle("About Glimpse MRI");
    auto* lay = new QVBoxLayout(&dlg);

    addAboutDescription(lay);

    auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok, &dlg);
    QObject::connect(bb, &QDialogButtonBox::accepted, &dlg, &QDialog::accept);
    lay->addWidget(bb);

    dlg.resize(560, 400);
    const int rc = dlg.exec();
    qDebug() << "[About][UI] showAboutDialog EXIT rc=" << rc;
}

void MainWindow::addAboutDescription(QVBoxLayout* layout)
{
    qDebug() << "[About][UI] addAboutDescription ENTER layout?" << (layout ? "YES" : "NO");
    if (!layout) {
        qWarning() << "[About][UI][WRN] addAboutDescription called with null layout";
        return;
    }

    const QString description =
        "<p><b>Glimpse MRI</b> is a high-performance MRI viewing and reconstruction application crafted for efficiency "
        "and designed to deliver powerful, reliable image handling. It supports <b>DICOM</b>, <b>ISMRMRD (HDF5)</b>, "
        "and <b>fastMRI</b> datasets, and can both view and write MRI images.</p>"

        "<p>Developed using the <b>Model-View-Controller (MVC)</b> architecture, <b>Glimpse MRI</b> combines "
        "<b>C++</b>, <b>CUDA</b>, and <b>Qt</b> for a clean, responsive interface while delegating compute-intensive work to a "
        "<b>High-Performance Heterogeneous custom MRI Engine</b>.</p>"

        "<p>The MRI engine targets modern CPUs and GPUs—with multi-threading via <b>OpenMP</b>, CPU-side Fourier transforms using "
        "<b>FFTW (Fastest Fourier Transform in the West)</b>, and GPU acceleration through custom <b>CUDA</b> kernels, "
        "<b>NVIDIA cuFFT (CUDA Fast Fourier Transform library)</b>, and "
        "<b>NVIDIA cuBLAS (CUDA Basic Linear Algebra Subprograms)</b>. "
        "This heterogeneous pipeline enables fast reconstruction and smooth, real-time interaction on supported hardware.</p>"

        "<p>Built and designed by Agustin Tortolero. Fueled by copious amounts of yerba mate.</p>"
        "<p><b>Contact</b>: agustin.tortolero@proton.me</p>"
        "<p><b>Source code</b>: <a href='https://github.com/agustinTortolero/glimpse_mri'>GitHub</a></p>";

    auto* browser = new QTextBrowser();
    browser->setObjectName("aboutDescription");
    browser->setOpenExternalLinks(true);
    browser->setReadOnly(true);
    browser->setStyleSheet("border: none; background: transparent;");
    browser->document()->setDefaultFont(QFont("Verdana", 10));
    browser->setHtml(description);

    layout->addWidget(browser);
    qDebug() << "[About][UI] addAboutDescription DONE widget=" << browser;
}
