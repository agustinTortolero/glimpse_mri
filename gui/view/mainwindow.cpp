
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
#include <QTabWidget>


#include <memory>
#include <iostream>
#include <algorithm>
#include <chrono>

#include <opencv2/imgproc.hpp>

#include "engine_api.h"
#include <QSysInfo>
#include <QLibraryInfo>
#include "build_info.hpp"

using namespace std::chrono_literals;

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent)
{
    std::cerr << "[DBG][View][Ctor] start\n";
    setAcceptDrops(true);


    applyGlobalDarkTheme();
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

    createMetadataDock();
    createHistogramDock();

    setInitialSize();
    showDragHint();
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
    label->installEventFilter(this);

    dock->setWidget(label);
    addDockWidget(Qt::RightDockWidgetArea, dock);

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

    setImage(m);

    if (m_histDock && m_histDock->isVisible() && m_histLabel) {
        const QSize area = m_histLabel->contentsRect().size();
        qDebug() << "[View][setImageCV8U] requestHistogramUpdate size=" << area;
        emit requestHistogramUpdate(area);
    }

    qDebug() << "[View][setImageCV8U] EXIT";
}

QWidget* MainWindow::createCentralArea()
{
    qDebug() << "[View][createCentralArea]";
    auto* central = new QWidget(this);
    auto* vlay    = new QVBoxLayout(central);
    vlay->setContentsMargins(8, 8, 8, 8);
    vlay->setSpacing(6);

    m_label = new QLabel(central);
    m_label->setAlignment(Qt::AlignCenter);
    m_label->setText("No image");
    m_label->installEventFilter(this);
    vlay->addWidget(m_label, 1);

    auto* row  = new QWidget(central);
    auto* hlay = new QHBoxLayout(row);
    hlay->setContentsMargins(0,0,0,0);
    hlay->setSpacing(8);

    m_sliceSlider = new QSlider(Qt::Horizontal, row);
    m_sliceSlider->setEnabled(false);
    m_sliceSlider->setRange(0, 0);
    m_sliceSlider->setValue(0);
    m_sliceSlider->setVisible(false);

    hlay->addWidget(m_sliceSlider, 1);
    row->setLayout(hlay);
    vlay->addWidget(row, 0);

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

    dock->setFloating(false);
    dock->hide();

    m_metaDock = dock;
    m_metaText = txt;

    m_metaDock->setProperty("stackedOnce", false);
    m_metaDock->setProperty("sizedOnce",   false);
    m_metaDock->setProperty("vSizedOnce",  false);

    qDebug() << "[View][Meta] Dock created (hidden by default)";
}

void MainWindow::initRightDocksIfNeeded()
{
    if (!(m_metaDock && m_histDock)) return;

    const bool firstInit = !m_metaDock->property("sizedOnce").toBool();
    if (!firstInit) return;

    m_metaDock->show();
    m_histDock->show();

    if (!m_metaDock->property("stackedOnce").toBool()) {
        splitDockWidget(m_metaDock, m_histDock, Qt::Vertical);
        m_metaDock->setProperty("stackedOnce", true);
        qDebug() << "[View][Dock] splitDockWidget(meta, hist, Vertical)";
    }

    const int halfW = qMax(120, width() / 2);
    QList<QDockWidget*> docksW { m_metaDock };
    QList<int>          sizesW { halfW };
    resizeDocks(docksW, sizesW, Qt::Horizontal);
    m_metaDock->setProperty("sizedOnce", true);
    qDebug() << "[View][Dock] resizeDocks Horizontal -> halfW=" << halfW << " winW=" << width();

    const int halfH = qMax(60, height() / 2);
    QList<QDockWidget*> docksH { m_metaDock, m_histDock };
    QList<int>          sizesH { halfH, halfH };
    resizeDocks(docksH, sizesH, Qt::Vertical);
    m_metaDock->setProperty("vSizedOnce", true);
    qDebug() << "[View][Dock] resizeDocks Vertical -> halfH=" << halfH << " winH=" << height();
}

void MainWindow::setInitialSize()
{
    resize(900, 571);
    qDebug() << "[View] Initial size set to" << width() << "x" << height();
}

bool MainWindow::eventFilter(QObject* obj, QEvent* ev)
{
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
        qDebug() << "[View][Hist] label resized -> requestHistogramUpdate";
        if (m_histDock && m_histDock->isVisible()) {
            const QSize area = m_histLabel->contentsRect().size();
            emit requestHistogramUpdate(area);
        }
        return false;
    }

    return QMainWindow::eventFilter(obj, ev);
}


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
    enableSliceSlider(0);

    if (m_metaDock && m_metaDock->isVisible()) m_metaDock->hide();
    if (m_histDock && m_histDock->isVisible()) m_histDock->hide();

    if (m_metaText)  m_metaText->setPlainText(QString());
    if (m_histLabel) m_histLabel->setText("No image");

    qDebug() << "[View] Docks hidden (no image)";
}



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

    const QString line = QString("Image %1x%2 | min=%3 max=%4")
                             .arg(m.cols)
                             .arg(m.rows)
                             .arg(minv, 0, 'f', 3)
                             .arg(maxv, 0, 'f', 3);

    appendMetadataLine(line);
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
    cv::Mat m = to8uMono(img8u);
    logMatStats(m);
    storeImage(m);
    updateMetadataForImage(m);
    repaintOnce();

    initRightDocksIfNeeded();
    if (m_histDock && m_histDock->isVisible() && m_histLabel) {
        const QSize area = m_histLabel->contentsRect().size();
        emit requestHistogramUpdate(area);
    }

    qDebug() << "[View] setImage EXIT";
}


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
        return m_img8;
    }
    cv::Mat shown = m_img8.clone();
    drawSliceOverlay(shown);
    return shown;
}

QImage MainWindow::toQImageOwned(const cv::Mat& m) const
{
    QImage q(m.data, m.cols, m.rows, static_cast<int>(m.step), QImage::Format_Grayscale8);
    return q.copy();
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
    refreshPixmap();
    qDebug() << "[View][Slider] state -> enabled=" << m_sliceSlider->isEnabled()
             << " visible=" << m_sliceSlider->isVisible()
             << " value=" << m_sliceSlider->value();
}

void MainWindow::setSliceIndex(int idx)
{
    if (!m_sliceSlider) return;
    QSignalBlocker block(m_sliceSlider);
    idx = std::clamp(idx, m_sliceSlider->minimum(), m_sliceSlider->maximum());
    m_sliceSlider->setValue(idx);
    qDebug() << "[View][Slider] setSliceIndex ->" << idx << "(signals blocked)";
    refreshPixmap();
}

void MainWindow::onSliderValueChanged(int v)
{
    qDebug() << "[View][Slider] valueChanged -> emit sliceChanged(" << v << ")";
    emit sliceChanged(v);
    refreshPixmap();
}

void MainWindow::onNegativeModeChanged(bool on)
{
    m_negativeMode = on;
    qDebug() << "[View] onNegativeModeChanged ->" << on;

    if (m_histDock && m_histDock->isVisible() && m_histLabel) {
        const QSize area = m_histLabel->contentsRect().size();
        emit requestHistogramUpdate(area);
    }
}


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
    qDebug() << "[UI][Menu] populateMenuForImage ENTER hasImg=" << hasImg
             << " hasMulti=" << hasMulti;

    // Save options (unchanged)
    out.saveSlice = menu.addAction("Save slice...");
    out.saveBatch = menu.addAction("Save batch...");
    menu.addSeparator();

    // View toggles
    out.negative  = menu.addAction("Negative");
    out.negative->setCheckable(true);
    out.negative->setChecked(m_negativeMode);
    out.negative->setEnabled(hasImg);

    QAction* histAct = menu.addAction("Histogram");
    histAct->setObjectName(QStringLiteral("action_histogram"));
    histAct->setCheckable(true);
    const bool histVisible = (m_histDock && m_histDock->isVisible());
    histAct->setChecked(histVisible);
    histAct->setEnabled(hasImg);

    QAction* infoAct = menu.addAction("Image Info");
    infoAct->setObjectName(QStringLiteral("action_image_info"));
    infoAct->setCheckable(true);
    const bool infoVisible = (m_metaDock && m_metaDock->isVisible());
    infoAct->setChecked(infoVisible);
    infoAct->setEnabled(hasImg);

    menu.addSeparator();
    out.startOver = menu.addAction("Start over");

    // Enable/disable per context
    if (out.saveSlice) out.saveSlice->setEnabled(hasImg);
    if (out.saveBatch) out.saveBatch->setEnabled(hasImg && hasMulti);

    qDebug() << "[UI][Menu] populateMenuForImage EXIT"
             << " saveSlice=" << (out.saveSlice ? "Y" : "N")
             << " saveBatch=" << (out.saveBatch ? "Y" : "N")
             << " neg=" << (out.negative ? "Y" : "N")
             << " startOver=" << (out.startOver ? "Y" : "N");
}


void MainWindow::onQuickSaveDicom()
{
    if (!m_hasImage || m_img8.empty()) {
        qWarning() << "[UI] onQuickSaveDicom: no image to save";
        return;
    }
    const QString autoPath; // empty => controller uses defaults
    qDebug() << "[UI] onQuickSaveDicom: emit requestSaveDICOM(<auto>)";
    emit requestSaveDICOM(autoPath);
    if (statusBar()) statusBar()->showMessage("DICOM quick-saved (default path).", 2000);
}



void MainWindow::applyContextSelection(QAction* chosen, const CtxMenuActions& acts)
{

    if (!chosen) { qDebug() << "[UI][Menu] no action selected"; return; }

    if (chosen->objectName() == QLatin1String("action_histogram")) {
        qDebug() << "[UI][Menu] Histogram toggled";
        if (!m_histDock) createHistogramDock();
        const bool show = !m_histDock->isVisible();
        if (show) {
            initRightDocksIfNeeded();
            m_histDock->show();
            m_histDock->raise();
            qDebug() << "[UI][Hist] dock -> SHOW";
            if (m_histLabel) {
                const QSize area = m_histLabel->contentsRect().size();
                emit requestHistogramUpdate(area);
            }
        } else {
            m_histDock->hide();
            qDebug() << "[UI][Hist] dock -> HIDE";
        }
        return;
    }

    if (chosen->objectName() == QLatin1String("action_image_info")) {
        qDebug() << "[UI][Menu] Image Info toggled";
        if (!m_metaDock) createMetadataDock();
        const bool show = !m_metaDock->isVisible();
        if (show) {
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

    if (chosen == acts.saveSlice) { qDebug() << "[UI][Menu] Save slice selected"; onSavePNG(); return; }
    if (chosen == acts.saveBatch) { qDebug() << "[UI][Menu] Save batch selected"; onSaveBatch(); return; }
    if (chosen == acts.negative)  { qDebug() << "[UI][Menu] Negative toggled -> emit requestApplyNegative()"; emit requestApplyNegative(); return; }
    if (chosen == acts.startOver) { qDebug() << "[UI][Menu] Start over selected -> emit startOverRequested()"; emit startOverRequested(); return; }
    if (chosen == acts.about)     { qDebug() << "[UI][Menu] About selected -> showAboutDialog()"; showAboutDialog(); return; }

    qDebug() << "[UI][Menu] applyContextSelection: unhandled action";
}


bool MainWindow::promptSingleSave(QString* outPath, SaveFmt* outFmt)
{
    const QString pics = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    const QString baseDir = (pics.isEmpty() ? QDir::homePath() : pics) + "/Glimpse";
    if (!QDir(baseDir).exists()) {
        const bool ok = QDir().mkpath(baseDir);
        qDebug() << "[Save][Single] mkpath(" << baseDir << ") ->" << (ok ? "OK" : "FAIL");
    }

    QFileDialog dlg(this, tr("Save slice"), baseDir);
    dlg.setAcceptMode(QFileDialog::AcceptSave);
    dlg.setFileMode(QFileDialog::AnyFile);

    QStringList filters;
    filters << "PNG (*.png)" << "DICOM (*.dcm)";
    dlg.setNameFilters(filters);
    dlg.selectNameFilter("PNG (*.png)");
    dlg.selectFile("image");

    if (!dlg.exec()) {
        qDebug() << "[Save][Single] user canceled";
        return false;
    }

    const QString chosenFilter = dlg.selectedNameFilter();
    QString fn = dlg.selectedFiles().isEmpty() ? QString() : dlg.selectedFiles().front();
    if (fn.isEmpty()) {
        qWarning() << "[Save][Single] empty filename after exec()";
        return false;
    }

    SaveFmt fmt = SaveFmt::PNG;
    const QString ext = QFileInfo(fn).suffix().toLower();
    if (ext == "png") fmt = SaveFmt::PNG;
    else if (ext == "dcm") fmt = SaveFmt::DICOM;
    else if (chosenFilter.contains("DICOM", Qt::CaseInsensitive)) fmt = SaveFmt::DICOM;
    else fmt = SaveFmt::PNG;

    if (fmt == SaveFmt::PNG && !fn.endsWith(".png", Qt::CaseInsensitive)) fn += ".png";
    if (fmt == SaveFmt::DICOM && !fn.endsWith(".dcm", Qt::CaseInsensitive)) fn += ".dcm";

    if (outPath) *outPath = fn;
    if (outFmt)  *outFmt  = fmt;

    qDebug() << "[Save][Single] chosen path=" << fn
             << " fmt=" << (fmt == SaveFmt::PNG ? "PNG" : "DICOM");
    return true;
}

void MainWindow::emitSingleSave(const QString& path, SaveFmt fmt)
{
    if (fmt == SaveFmt::PNG) {
        qDebug() << "[Save][Single] emit requestSavePNG ->" << path;
        emit requestSavePNG(path);
    } else {
        qDebug() << "[Save][Single] emit requestSaveDICOM ->" << path;
        emit requestSaveDICOM(path);
    }
}

void MainWindow::onSavePNG()
{
    if (!m_hasImage || m_img8.empty()) {
        qWarning() << "[UI] onSavePNG(unified): no image to save";
        return;
    }
    QString path; SaveFmt fmt;
    if (!promptSingleSave(&path, &fmt)) return;
    emitSingleSave(path, fmt);
}

bool MainWindow::canBatchSave() const
{
    if (!m_hasImage || m_img8.empty() || !m_sliceSlider) return false;
    const int S = m_sliceSlider->maximum() - m_sliceSlider->minimum() + 1;
    return (m_sliceSlider->isEnabled() && S >= 2);
}

bool MainWindow::promptBatchDestination(QString* outBasePath, SaveFmt* outFmt)
{
    const QString pics = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    const QString baseDir = (pics.isEmpty() ? QDir::homePath() : pics) + "/Glimpse";
    QDir().mkpath(baseDir);

    QFileDialog dlg(this, tr("Save batch (all slices)"), baseDir);
    dlg.setAcceptMode(QFileDialog::AcceptSave);
    dlg.setFileMode(QFileDialog::AnyFile);

    QStringList filters; filters << "PNG (*.png)" << "DICOM (MR series) (*.dcm)";
    dlg.setNameFilters(filters);
    dlg.selectNameFilter("PNG (*.png)");
    dlg.selectFile("series");

    if (!dlg.exec()) {
        qDebug() << "[Save][Batch] user canceled";
        return false;
    }

    const QString chosenFilter = dlg.selectedNameFilter();
    QString basePath = dlg.selectedFiles().isEmpty() ? QString() : dlg.selectedFiles().front();
    if (basePath.isEmpty()) {
        qWarning() << "[Save][Batch] empty basePath after exec()";
        return false;
    }

    SaveFmt fmt = SaveFmt::PNG;
    const QString extTyped = QFileInfo(basePath).suffix().toLower();
    if (extTyped == "png")       fmt = SaveFmt::PNG;
    else if (extTyped == "dcm")  fmt = SaveFmt::DICOM_SERIES;
    else if (chosenFilter.contains("DICOM", Qt::CaseInsensitive)) fmt = SaveFmt::DICOM_SERIES;
    else                        fmt = SaveFmt::PNG;

    const QString finalExt = (fmt == SaveFmt::PNG) ? ".png" : ".dcm";
    if (!basePath.endsWith(finalExt, Qt::CaseInsensitive)) basePath += finalExt;

    if (outBasePath) *outBasePath = basePath;
    if (outFmt)      *outFmt      = fmt;

    qDebug() << "[Save][Batch] basePath=" << basePath
             << " fmt=" << (fmt == SaveFmt::PNG ? "PNG-many" : "DICOM MR series-many");
    return true;
}

int MainWindow::computeIndexPadding(int S) const
{
    int pad = 2;
    if      (S >= 10000) pad = 5;
    else if (S >= 1000 ) pad = 4;
    else if (S >= 100  ) pad = 3;
    return pad;
}

void MainWindow::saveBatchPNGSlices(const QString& basePath, int S, int pad)
{
    const QFileInfo fi(basePath);
    const QString dir   = fi.absolutePath();
    const QString stem  = fi.completeBaseName();
    const QString ext   = fi.suffix().toLower();

    int saved = 0;
    const int minIdx = m_sliceSlider->minimum();
    for (int k = 0; k < S; ++k) {
        const int idx = minIdx + k;
        const QString idxStr = QString("%1").arg(k + 1, pad, 10, QChar('0'));
        const QString outPath = dir + "/" + stem + "_" + idxStr + "." + ext;

        { QSignalBlocker block(*m_sliceSlider); m_sliceSlider->setValue(idx); }
        emit sliceChanged(idx);
        QCoreApplication::processEvents(QEventLoop::AllEvents, 8);

        qDebug() << "[Save][Batch-PNG] emit requestSavePNG ->" << outPath;
        emit requestSavePNG(outPath);
        ++saved;
    }
    if (statusBar()) statusBar()->showMessage(
            QString("Saved %1 PNG file(s) to %2").arg(saved).arg(dir), 3000);
    qDebug() << "[Save][Batch-PNG] EXIT saved=" << saved;
}

bool MainWindow::promptDicomSeriesGeometry(double* px, double* py, double* sth, double* sbs,
                                           QVector<double>* iop6, QVector<double>* ipp0)
{
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

    auto* spx = mkDsb(1.0, 0.01, 100.0, 3);
    auto* spy = mkDsb(1.0, 0.01, 100.0, 3);
    auto* sthick = mkDsb(1.0, 0.01, 200.0, 3);
    auto* sbetween = mkDsb(1.0, 0.00, 200.0, 3);

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
    lay->addRow("Slice Thickness:", sthick);
    lay->addRow("Spacing Between Slices:", sbetween);
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

    const bool ok = geo.exec();
    if (!ok) {
        qDebug() << "[Save][Batch-DICOM] geometry dialog canceled";
        return false;
    }

    if (px) *px = spx->value();
    if (py) *py = spy->value();
    if (sth) *sth = sthick->value();
    if (sbs) *sbs = sbetween->value();
    if (iop6) *iop6 = QVector<double>{ r1->value(), r2->value(), r3->value(),
                                c1->value(), c2->value(), c3->value() };
    if (ipp0) *ipp0 = QVector<double>{ x0->value(), y0->value(), z0->value() };

    qDebug() << "[Save][Batch-DICOM] geometry:"
             << "px=" << spx->value() << "py=" << spy->value()
             << "thick=" << sthick->value() << "spacing=" << sbetween->value()
             << "IOP6=" << (iop6 ? *iop6 : QVector<double>{})
             << "IPP0=" << (ipp0 ? *ipp0 : QVector<double>{});
    return true;
}

void MainWindow::emitDicomSeries(const QString& basePath, double px, double py, double sth, double sbs,
                                 const QVector<double>& iop6, const QVector<double>& ipp0)
{
    qDebug() << "[Save][Batch-DICOM] emit requestSaveDICOMSeriesMR ->" << basePath;
    emit requestSaveDICOMSeriesMR(basePath, px, py, sth, sbs, iop6, ipp0);
}

void MainWindow::onSaveBatch()
{
    if (!canBatchSave()) {
        qWarning() << "[UI] onSaveBatch: only one slice or no slider -> disabled";
        return;
    }

    QString basePath; SaveFmt fmt;
    if (!promptBatchDestination(&basePath, &fmt)) return;

    const int S = m_sliceSlider->maximum() - m_sliceSlider->minimum() + 1;
    if (fmt == SaveFmt::PNG) {
        const int pad = computeIndexPadding(S);
        saveBatchPNGSlices(basePath, S, pad);
        return;
    }

    // Quick DICOM series (no geometry dialog)
    double px  = 1.0, py  = 1.0;     // default PixelSpacing
    double sth = 1.0, sbs = 1.0;     // default SliceThickness and SpacingBetweenSlices
    QVector<double> iop6  = { 1, 0, 0, 0, 1, 0 };  // default IOP (rows = X, cols = Y)
    QVector<double> ipp0  = { 0, 0, 0 };          // default IPP for slice #1

    qDebug() << "[Save][Batch-DICOM] QUICK no-geometry:"
             << "px=" << px << "py=" << py
             << "thick=" << sth << "spacing=" << sbs
             << "IOP6=" << iop6 << "IPP0=" << ipp0;

    emitDicomSeries(basePath, px, py, sth, sbs, iop6, ipp0);


    const QString dir = QFileInfo(basePath).absolutePath();
    if (statusBar()) statusBar()->showMessage(
            QString("Writing MR series to %1 (N files)").arg(dir), 2500);
}

void MainWindow::showDragHint()
{
    if (!m_label) return;
    m_label->setText("Drag DICOM or HDF5\n(.dcm, .h5, .hdf5)\n");
    qDebug() << "[DnD][View] drag hint shown";
}

void MainWindow::clearDragHint()
{
    if (!m_label) return;
    if (m_hasImage) {
        qDebug() << "[DnD][View] drag hint cleared (image present)";
    } else {
        m_label->clear();
        qDebug() << "[DnD][View] drag hint cleared (no image yet)";
    }
}

void MainWindow::drawSliceOverlay(cv::Mat& img8)
{
    if (img8.empty()) return;
    if (!m_sliceSlider || !m_sliceSlider->isEnabled()) return;

    const int S = m_sliceSlider->maximum() - m_sliceSlider->minimum() + 1;
    if (S < 2) return;

    const int s = m_sliceSlider->value();

    cv::Mat c3;
    cv::cvtColor(img8, c3, cv::COLOR_GRAY2BGR);

    const std::string txt = "Slice " + std::to_string(s + 1) + "/" + std::to_string(S);

    const double fontScale = 0.6;
    const int    thin      = 1;
    int baseline = 0;
    cv::Size ts = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, fontScale, thin, &baseline);

    cv::Point org(12, 24);
    cv::Rect box(org.x - 8, org.y - ts.height - 8, ts.width + 16, ts.height + baseline + 16);
    box &= cv::Rect(0, 0, c3.cols, c3.rows);

    if (box.area() > 0) {
        cv::Mat roi = c3(box);
        cv::Mat overlay = roi.clone();
        cv::rectangle(overlay, cv::Rect(0, 0, roi.cols, roi.rows), cv::Scalar(0, 0, 0), cv::FILLED);
        const double alpha = 0.35;
        cv::addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi);
    }

    cv::putText(c3, txt, org, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(c3, txt, org, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    cv::cvtColor(c3, img8, cv::COLOR_BGR2GRAY);

    qDebug() << "[View][Overlay]" << QString::fromStdString(txt)
             << "box=" << box.width << "x" << box.height
             << "alpha=0.35";
}


void MainWindow::addVersionTab(QTabWidget* tabs)
{
    qDebug() << "[UI][About] addVersionTab() ENTER";

    // From .pro DEFINES (or fallback from build_info.hpp)
    const QString guiVersion     = QString::fromLatin1(GUI_VERSION_STR);
    const QString buildType      = QString::fromLatin1(BUILD_TYPE_STR);
    const QString gitSha         = QString::fromLatin1(GIT_SHA_STR);
    const QString gitDescribe    = QString::fromLatin1(GIT_DESCRIBE_STR);
    const QString buildTimestamp = QString::fromLatin1(BUILD_TIMESTAMP_STR);
    const QString qtQmakeVersion = QString::fromLatin1(QT_BUILD_VERSION_STR);
    const QString qmakeVersion   = QString::fromLatin1(QMAKE_BUILD_VERSION_STR);
    const QString qmakePath      = QString::fromLatin1(QMAKE_PATH_STR);

    qDebug() << "[UI][About] guiVersion =" << guiVersion;
    qDebug() << "[UI][About] buildType  =" << buildType;
    qDebug() << "[UI][About] git        =" << gitDescribe << "(" << gitSha << ")";
    qDebug() << "[UI][About] builtAt    =" << buildTimestamp;
    qDebug() << "[UI][About] qt(qmake)  =" << qtQmakeVersion;
    qDebug() << "[UI][About] qmake      =" << qmakeVersion;
    qDebug() << "[UI][About] qmakePath  =" << qmakePath;

    // Engine version
    const char* verCStr = engine_version();
    QString engineVersion = verCStr ? QString::fromUtf8(verCStr)
                                    : QStringLiteral("Unknown");

    qDebug() << "[UI][About] engine_version() ->" << engineVersion;

    // Qt runtime/compile info
    const QString qtCompile = QString::fromLatin1(QT_VERSION_STR);
    const QString qtRuntime = QString::fromLatin1(qVersion());
    const QString qtLibInfo = QLibraryInfo::version().toString();

    qDebug() << "[UI][About] Qt compile-time =" << qtCompile;
    qDebug() << "[UI][About] Qt runtime     =" << qtRuntime;
    qDebug() << "[UI][About] Qt (QLibraryInfo)=" << qtLibInfo;

    // OS / ABI
    const QString osPretty = QSysInfo::prettyProductName();
    const QString kernel   = QString("%1 %2").arg(QSysInfo::kernelType(),
                                                QSysInfo::kernelVersion());
    const QString arch = QSysInfo::currentCpuArchitecture();
    const QString abi  = QSysInfo::buildAbi();

    qDebug() << "[UI][About] OS=" << osPretty << "Kernel=" << kernel
             << "Arch=" << arch << "ABI=" << abi;

    // Build the text
    QString versionText;
    versionText += QString("GUI: %1\n").arg(guiVersion);
    versionText += QString("MRI Engine: %1\n").arg(engineVersion);
    versionText += "\n";

    versionText += QString("Build type: %1\n").arg(buildType);
    versionText += QString("Git commit: %1\n").arg(gitSha);
    versionText += QString("Git describe: %1\n").arg(gitDescribe);
    versionText += QString("Built at: %1\n").arg(buildTimestamp);
    versionText += "\n";

    versionText += QString("Qt (compile): %1\n").arg(qtCompile);
    versionText += QString("Qt (runtime): %1\n").arg(qtRuntime);
    versionText += QString("Qt (QLibraryInfo): %1\n").arg(qtLibInfo);
    versionText += QString("Qt (qmake): %1\n").arg(qtQmakeVersion);
    versionText += QString("qmake: %1\n").arg(qmakeVersion);
    versionText += QString("qmake path: %1\n").arg(qmakePath);
    versionText += "\n";

    versionText += QString("OS: %1\n").arg(osPretty);
    versionText += QString("Kernel: %1\n").arg(kernel);
    versionText += QString("Arch: %1\n").arg(arch);
    versionText += QString("ABI: %1\n").arg(abi);

    qDebug() << "[UI][About] Version tab text to display:\n" << versionText;

    auto* browser = new QTextBrowser(tabs);
    browser->setFrameShape(QFrame::NoFrame);
    browser->setReadOnly(true);
    browser->setText(versionText);

    browser->setStyleSheet(
        "QTextBrowser { "
        "  border: none; "
        "  background-color: #7a7a7a; "
        "  color: #202020; "
        "}"
        );

    tabs->addTab(browser, tr("Version"));
    qDebug() << "[UI][About] addVersionTab() EXIT";
}

void MainWindow::addHardwareTab(QTabWidget* tabs)
{
    qDebug() << "[UI][About] addHardwareTab() ENTER";

    engine_init_info_t info{};
    engine_get_last_init_info(&info);

    // Smoke-test: print everything to the console
    qDebug() << "[UI][About][HW] struct_size       =" << info.struct_size;
    qDebug() << "[UI][About][HW] cuda_compiled     =" << info.cuda_compiled;
    qDebug() << "[UI][About][HW] has_cuda          =" << info.has_cuda;
    qDebug() << "[UI][About][HW] force_cpu         =" << info.force_cpu;
    qDebug() << "[UI][About][HW] cuda_device_index =" << info.cuda_device_index;
    qDebug() << "[UI][About][HW] cc                ="
             << info.cuda_cc_major << "." << info.cuda_cc_minor;
    qDebug() << "[UI][About][HW] mem_gb            =" << info.cuda_mem_gb;
    qDebug() << "[UI][About][HW] omp_enabled       =" << info.omp_enabled;
    qDebug() << "[UI][About][HW] omp_max_threads   =" << info.omp_max_threads;
    qDebug() << "[UI][About][HW] cuda_name         ="
             << (info.cuda_name ? info.cuda_name : "(null)");

    QString hwText;

    // Backend summary
    if (info.cuda_compiled && info.has_cuda && !info.force_cpu) {
        const QString cudaName =
            info.cuda_name ? QString::fromUtf8(info.cuda_name) : QString();

        hwText += QString("Backend: CUDA device %1 (%2)\n")
                      .arg(info.cuda_device_index)
                      .arg(cudaName);
        hwText += QString("Compute capability: %1.%2\n")
                      .arg(info.cuda_cc_major)
                      .arg(info.cuda_cc_minor);
        hwText += QString("Device memory: %1 GB\n")
                      .arg(info.cuda_mem_gb);
    } else {
        if (!info.cuda_compiled) {
            hwText += QStringLiteral(
                "Backend: CPU only (CUDA backend not compiled)\n");
        } else if (info.force_cpu) {
            hwText += QStringLiteral(
                "Backend: CPU (forced CPU mode)\n");
        } else if (!info.has_cuda) {
            hwText += QStringLiteral(
                "Backend: CPU (no usable CUDA device found)\n");
        } else {
            hwText += QStringLiteral(
                "Backend: CPU backend (unspecified)\n");
        }
    }

    hwText += "\n";
    hwText += QString("CUDA compiled: %1\n")
                  .arg(info.cuda_compiled ? "Yes" : "No");
    hwText += QString("Has CUDA device: %1\n")
                  .arg(info.has_cuda ? "Yes" : "No");
    hwText += QString("Force CPU: %1\n")
                  .arg(info.force_cpu ? "Yes" : "No");
    hwText += QString("CUDA device index: %1\n")
                  .arg(info.cuda_device_index);
    hwText += QString("CUDA device name: %1\n")
                  .arg(info.cuda_name ? QString::fromUtf8(info.cuda_name)
                                      : QStringLiteral("(unknown)"));
    hwText += QString("Compute capability: %1.%2\n")
                  .arg(info.cuda_cc_major)
                  .arg(info.cuda_cc_minor);
    hwText += QString("Device memory: %1 GB\n")
                  .arg(info.cuda_mem_gb);
    hwText += QString("OpenMP enabled: %1\n")
                  .arg(info.omp_enabled ? "Yes" : "No");
    hwText += QString("OpenMP max threads: %1\n")
                  .arg(info.omp_max_threads);

    qDebug() << "[UI][About] Hardware tab text to display:" << hwText;

    auto* browser = new QTextBrowser(tabs);
    browser->setFrameShape(QFrame::NoFrame);
    browser->setReadOnly(true);
    browser->setText(hwText);

    browser->setStyleSheet(
        "QTextBrowser { "
        "  border: none; "
        "  background-color: #7a7a7a; "
        "  color: #202020; "
        "}"
        );

    tabs->addTab(browser, tr("Hardware"));
    qDebug() << "[UI][About] addHardwareTab() EXIT";
}


void MainWindow::showAboutDialog()
{
    qDebug() << "[About][UI] showAboutDialog ENTER";

    QDialog dlg(this);
    dlg.setWindowTitle("About Glimpse MRI");

    auto* vbox = new QVBoxLayout(&dlg);

    // --- Tabs container ---
    auto* tabs = new QTabWidget(&dlg);
    tabs->setObjectName("aboutTabs");
    qDebug() << "[About][UI] Created QTabWidget for About dialog";

    // --- Overview tab (was: About) ---
    auto* overviewPage   = new QWidget(tabs);
    auto* overviewLayout = new QVBoxLayout(overviewPage);
    overviewLayout->setContentsMargins(8, 8, 8, 8);
    overviewLayout->setSpacing(8);

    addAboutDescription(overviewLayout);
    addAboutBadges(overviewLayout);

    overviewPage->setLayout(overviewLayout);
    tabs->addTab(overviewPage, tr("Overview"));
    qDebug() << "[About][UI] Added 'Overview' tab";

    // --- Tools tab ---
    auto* toolsPage   = new QWidget(tabs);
    auto* toolsLayout = new QVBoxLayout(toolsPage);
    toolsLayout->setContentsMargins(8, 8, 8, 8);
    toolsLayout->setSpacing(8);

    addToolsTab(toolsLayout);

    toolsPage->setLayout(toolsLayout);
    tabs->addTab(toolsPage, tr("Tools"));
    qDebug() << "[About][UI] Added 'Tools' tab";

    // --- Disclaimer tab ---
    auto* discPage   = new QWidget(tabs);
    auto* discLayout = new QVBoxLayout(discPage);
    discLayout->setContentsMargins(8, 8, 8, 8);
    discLayout->setSpacing(8);

    addDisclaimerTab(discLayout);

    discPage->setLayout(discLayout);
    tabs->addTab(discPage, tr("Disclaimer"));
    qDebug() << "[About][UI] Added 'Disclaimer' tab";

    // --- Version tab ---
    addVersionTab(tabs);

    // --- Hardware tab ---
    addHardwareTab(tabs);

    vbox->addWidget(tabs);

    // --- Buttons ---
    auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok, &dlg);
    QObject::connect(bb, &QDialogButtonBox::accepted, &dlg, &QDialog::accept);
    vbox->addWidget(bb);

    dlg.resize(600, 420);
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

    static const char* kAboutHtml = R"(
<div style='text-align: justify;'>
<p><b>Glimpse MRI</b> is a high-performance MRI reconstruction application with image viewing and writing capabilities. It supports <b>DICOM</b>, <b>ISMRMRD (HDF5)</b>, and <b>fastMRI</b> datasets.</p>

<p>Built with <b>C++</b>, <b>CUDA</b>, and <b>Qt</b>, <b>Glimpse MRI</b> delivers a clean, responsive interface while offloading heavy computation to a custom <b>high-performance, heterogeneous MRI engine</b>.</p>

<p>The engine targets modern CPUs and GPUs multi-threading with <b>OpenMP</b>, CPU-side Fourier transforms with <b>FFTW (Fastest Fourier Transform in the West)</b>, and GPU acceleration via custom <b>CUDA</b> kernels plus <b>NVIDIA cuFFT</b> and <b>NVIDIA cuBLAS</b>, enabling fast reconstruction and smooth, real-time interaction.</p>

<p>Designed and built by Agustin Tortolero.</p>
<p><b>Source code</b>: <a href='https://github.com/agustinTortolero/glimpse_mri'>github.com/agustinTortolero/glimpse_mri</a></p>
</div>
)";

    const QString description = QString::fromUtf8(kAboutHtml);

    auto* browser = new QTextBrowser();
    browser->setObjectName("aboutDescription");
    browser->setOpenExternalLinks(true);
    browser->setReadOnly(true);


    browser->setStyleSheet(
        "QTextBrowser { "
        "  border: none; "
        "  background-color: #7a7a7a; "
        "  color: #202020; "
        "}"
        );




    browser->document()->setDefaultFont(QFont("Verdana", 10));

    // Load HTML first
    browser->setHtml(description);
    qDebug() << "[About][UI] HTML loaded into QTextBrowser";

    // Force justify as fallback
    QTextOption opt = browser->document()->defaultTextOption();
    opt.setAlignment(Qt::AlignJustify);
    browser->document()->setDefaultTextOption(opt);
    qDebug() << "[About][UI] Applied document default alignment: AlignJustify";

    layout->addWidget(browser);
    qDebug() << "[About][UI] addAboutDescription DONE widget=" << browser;
}


void MainWindow::setHistogramImage(const QImage& img, const QString& tooltip)
{
    if (!m_histLabel) {
        qWarning() << "[View][Hist] setHistogramImage: label missing";
        return;
    }
    if (img.isNull()) {
        qWarning() << "[View][Hist] setHistogramImage: null QImage; will keep previous content";
        return;
    }
    m_histLabel->setPixmap(QPixmap::fromImage(img));
    if (!tooltip.isEmpty())
        m_histLabel->setToolTip(tooltip);
    qDebug() << "[View][Hist] setHistogramImage: applied pixmap"
             << " size=" << img.width() << "x" << img.height();
}

void MainWindow::addAboutBadges(QVBoxLayout* layout)
{
    qDebug() << "[About][UI] addAboutBadges ENTER layout?" << (layout ? "YES" : "NO");
    if (!layout) {
        qWarning() << "[About][UI][WRN] addAboutBadges called with null layout";
        return;
    }

    auto* row = new QWidget(this);
    auto* h   = new QHBoxLayout(row);
    h->setContentsMargins(0,0,0,0);
    h->setSpacing(16);

    struct Logo { const char* res; const char* alt; double scale; };

    // Order: C++ → NVIDIA → Qt → OpenCV → OpenMP (OpenMP scaled to 90%)
    const Logo logos[] = {
        {":/img/badges/Cpp_logo.png",    "C++",     1.00},
        {":/img/badges/Nvidia_logo.png", "NVIDIA",  1.00},
        {":/img/badges/Qt_logo.png",     "Qt",      1.00},
        {":/img/badges/OpenCV_logo.png", "OpenCV",  1.00},
        {":/img/badges/OpenMP_logo.png", "OpenMP",  0.90},  // -10%
    };

    const qreal dpr   = this->devicePixelRatioF();
    const int   baseH = (dpr > 1.0) ? 28 : 24;
    const int   targetH = int(std::lround(baseH * 1.5));  // your global +50%

    qDebug() << "[About][Logo] dpr=" << dpr << " baseH=" << baseH << " targetH=" << targetH << "(before per-logo scale)";

    auto addLogo = [&](const Logo& L){
        const QString path = QString::fromUtf8(L.res);
        const int thisH = std::max(1, int(std::lround(targetH * L.scale)));

        qDebug() << "[About][Logo] loading" << L.alt << "path=" << path
                 << " scale=" << L.scale << " thisH=" << thisH;

        QPixmap pm;
        const bool ok = pm.load(path);

        auto* lbl = new QLabel(row);
        lbl->setToolTip(QString::fromUtf8(L.alt));
        lbl->setMinimumHeight(thisH);
        lbl->setAlignment(Qt::AlignVCenter | Qt::AlignLeft);

        if (!ok || pm.isNull()) {
            qWarning() << "[About][Logo][WRN] missing badge resource:" << path;
            lbl->setText(QString::fromUtf8(L.alt));
            lbl->setStyleSheet("color: palette(mid);");
        } else {
            if (pm.height() != thisH) {
                pm = pm.scaledToHeight(thisH, Qt::SmoothTransformation);
            }
            lbl->setPixmap(pm);
        }

        h->addWidget(lbl, 0, Qt::AlignVCenter);
        qDebug() << "[About][Logo] added" << L.alt << " ok=" << ok << " finalH=" << thisH;
    };

    for (const auto& L : logos) addLogo(L);

    row->setLayout(h);
    layout->addWidget(row);

    QStringList order; for (const auto& L : logos) order << L.alt;
    qDebug() << "[About][UI] addAboutBadges DONE; order=" << order.join(" -> ");
}


void MainWindow::addDisclaimerTab(QVBoxLayout* layout)
{
    qDebug() << "[About][UI] addDisclaimerTab ENTER layout?" << (layout ? "YES" : "NO");
    if (!layout) {
        qWarning() << "[About][UI][WRN] addDisclaimerTab called with null layout";
        return;
    }

    static const char* kDisclaimerHtml = R"(
<div style='text-align: justify;'>
<p><b>Non-clinical software:</b> Glimpse MRI is intended solely for research, testing, and software development. It is <b>not a medical device</b>.</p>

<p>This application must not be used for diagnosis, treatment decisions, or any clinical workflow involving patients. Any images, reconstructions, or measurements produced are for demonstration and technical evaluation only.</p>

<p>No guarantee is made regarding the correctness, completeness, or suitability of the results for clinical purposes. By using this software, you acknowledge that it is provided &quot;as is&quot;, without any warranty of any kind.</p>
</div>
)";

    const QString html = QString::fromUtf8(kDisclaimerHtml);

    auto* browser = new QTextBrowser();
    browser->setObjectName("aboutDisclaimer");
    browser->setOpenExternalLinks(true);
    browser->setReadOnly(true);

    // Same color scheme as Overview tab
    browser->setStyleSheet(
        "QTextBrowser { "
        "  border: none; "
        "  background-color: #7a7a7a; "
        "  color: #202020; "
        "}"
        );

    browser->document()->setDefaultFont(QFont("Verdana", 10));

    browser->setHtml(html);
    qDebug() << "[About][UI] Disclaimer HTML loaded into QTextBrowser";

    QTextOption opt = browser->document()->defaultTextOption();
    opt.setAlignment(Qt::AlignJustify);
    browser->document()->setDefaultTextOption(opt);
    qDebug() << "[About][UI] Applied document default alignment: AlignJustify (Disclaimer)";

    layout->addWidget(browser);
    qDebug() << "[About][UI] addDisclaimerTab DONE widget=" << browser;
}


void MainWindow::addToolsTab(QVBoxLayout* layout)
{
    qDebug() << "[About][UI] addToolsTab ENTER layout?" << (layout ? "YES" : "NO");
    if (!layout) {
        qWarning() << "[About][UI][WRN] addToolsTab called with null layout";
        return;
    }

    static const char* kToolsHtml = R"(
<div style='text-align: justify;'>
  <h3>Core technologies</h3>
  <ul>
    <li><b>C++</b> (modern C++ for core engine and GUI)</li>
    <li><b>CUDA</b> for GPU-accelerated MRI reconstruction</li>
    <li><b>Qt 6</b> for the cross-platform GUI</li>
    <li><b>OpenMP</b> for multi-threaded CPU execution</li>
    <li><b>FFTW</b> (Fastest Fourier Transform in the West) for CPU-side FFTs</li>
    <li><b>NVIDIA cuFFT</b> and <b>NVIDIA cuBLAS</b> for GPU FFTs and BLAS operations</li>
  </ul>

  <h3>Data formats &amp; imaging libraries</h3>
  <ul>
    <li><b>DICOM</b> via a dedicated DICOM I/O library</li>
    <li><b>ISMRMRD (HDF5)</b> for raw MRI acquisitions</li>
    <li><b>fastMRI</b> single-coil datasets</li>
    <li><b>HDF5</b> and <b>HDF5 C++</b> APIs</li>
    <li><b>pugixml</b> for XML metadata handling</li>
  </ul>

  <h3>Development tools</h3>
  <ul>
    <li><b>Microsoft Visual Studio 2022</b> (MSVC toolchain, engine build)</li>
    <li><b>Qt Creator</b> (Qt 6 project, GUI build and deployment)</li>
    <li><b>qmake</b> for build configuration and project orchestration</li>
  </ul>

</div>
)";

    const QString html = QString::fromUtf8(kToolsHtml);

    auto* browser = new QTextBrowser();
    browser->setObjectName("aboutTools");
    browser->setOpenExternalLinks(true);
    browser->setReadOnly(true);

    // Same color scheme as Overview tab
    browser->setStyleSheet(
        "QTextBrowser { "
        "  border: none; "
        "  background-color: #7a7a7a; "
        "  color: #202020; "
        "}"
        );

    browser->document()->setDefaultFont(QFont("Verdana", 10));

    browser->setHtml(html);
    qDebug() << "[About][UI] Tools HTML loaded into QTextBrowser";

    QTextOption opt = browser->document()->defaultTextOption();
    opt.setAlignment(Qt::AlignJustify);
    browser->document()->setDefaultTextOption(opt);
    qDebug() << "[About][UI] Applied document default alignment: AlignJustify (Tools)";

    layout->addWidget(browser);
    qDebug() << "[About][UI] addToolsTab DONE widget=" << browser;
}



void MainWindow::applyDarkTheme()
{
    qDebug() << "[Theme][UI] Applying dark gray theme...";

    // Base colors
    QString base   = "#2b2b2b";   // dark gray
    QString panel  = "#3a3a3a";   // slightly lighter gray
    QString hover  = "#444444";
    QString text   = "#f0f0f0";
    QString border = "#555555";

    QString css = QString(R"(
        QMainWindow {
            background-color: %1;
        }
        QWidget {
            background-color: %1;
            color: %4;
        }
        QLabel {
            background-color: %1;
            color: %4;
        }
        QMenuBar, QMenu {
            background-color: %2;
            color: %4;
        }
        QMenu::item:selected {
            background-color: %3;
        }
        QStatusBar {
            background-color: %2;
            color: %4;
        }
        QPlainTextEdit, QTextEdit, QTextBrowser {
            background-color: %2;
            border: 1px solid %5;
            color: %4;
        }
        QDockWidget {
            background-color: %2;
            titlebar-close-icon: url(none);
            titlebar-normal-icon: url(none);
        }
        QDockWidget::title {
            background-color: %3;
            color: %4;
            padding: 4px;
        }
        QSlider::groove:horizontal {
            background: %5;
            height: 6px;
        }
        QSlider::handle:horizontal {
            background: %4;
            width: 14px;
            border-radius: 3px;
        }
    )")
                      .arg(base)   // %1
                      .arg(panel)  // %2
                      .arg(hover)  // %3
                      .arg(text)   // %4
                      .arg(border) // %5
        ;

    qApp->setStyleSheet(css);
    qDebug() << "[Theme][UI] Dark theme applied.";
}


void MainWindow::applyGlobalDarkTheme()
{
    qDebug() << "[Theme][UI] Applying global dark stylesheet";

    const QString css = R"(
        /* Base widget colors */
        QWidget {
            background-color: #262626;
            color: #f0f0f0;
        }

        QMainWindow {
            background-color: #262626;
        }

        /* Labels */
        QLabel {
            background-color: transparent;
            color: #f0f0f0;
        }

        /* Buttons */
        QPushButton, QToolButton {
            background-color: #3a3a3a;
            color: #f0f0f0;
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 4px 10px;
        }
        QPushButton:hover, QToolButton:hover {
            background-color: #4a4a4a;
        }
        QPushButton:pressed, QToolButton:pressed {
            background-color: #555555;
        }
        QPushButton:disabled, QToolButton:disabled {
            background-color: #333333;
            color: #888888;
            border-color: #444444;
        }

        /* Dialog buttons (OK, Cancel, etc.) */
        QDialogButtonBox QPushButton {
            min-width: 70px;
        }

        /* Menus */
        QMenuBar {
            background-color: #303030;
            color: #f0f0f0;
        }
        QMenuBar::item {
            background: transparent;
            padding: 4px 8px;
        }
        QMenuBar::item:selected {
            background: #404040;
        }
        QMenu {
            background-color: #303030;
            color: #f0f0f0;
            border: 1px solid #505050;
        }
        QMenu::item:selected {
            background-color: #4a4a4a;
        }

        /* Tabs */
        QTabWidget::pane {
            border: 1px solid #505050;
            background: #262626;
        }
        QTabBar::tab {
            background: #333333;
            color: #f0f0f0;
            padding: 6px 10px;
            border: 1px solid #505050;
            border-bottom: none;
        }
        QTabBar::tab:selected {
            background: #444444;
        }
        QTabBar::tab:hover {
            background: #505050;
        }

        /* Dock widgets */
        QDockWidget {
            background-color: #262626;
            titlebar-close-icon: url(none);
            titlebar-normal-icon: url(none);
        }
        QDockWidget::title {
            background-color: #333333;
            color: #f0f0f0;
            padding: 4px;
        }

        /* Status bar */
        QStatusBar {
            background-color: #303030;
            color: #d0d0d0;
        }

        /* Text widgets */
        QPlainTextEdit,
        QTextEdit,
        QTextBrowser,
        QLineEdit,
        QSpinBox,
        QDoubleSpinBox,
        QComboBox {
            background-color: #333333;
            color: #f0f0f0;
            border: 1px solid #555555;
            selection-background-color: #666666;
            selection-color: #ffffff;
        }

        /* ComboBox popup */
        QComboBox QAbstractItemView {
            background-color: #333333;
            color: #f0f0f0;
            selection-background-color: #4a4a4a;
        }

        /* Sliders */
        QSlider::groove:horizontal {
            background: #3a3a3a;
            height: 6px;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #f0f0f0;
            width: 14px;
            border-radius: 3px;
            margin: -4px 0; /* center handle */
        }

        /* Scrollbars */
        QScrollBar:vertical {
            background: #2a2a2a;
            width: 12px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #555555;
            min-height: 20px;
        }
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            background: #2a2a2a;
            height: 0;
        }
        QScrollBar:horizontal {
            background: #2a2a2a;
            height: 12px;
            margin: 0;
        }
        QScrollBar::handle:horizontal {
            background: #555555;
            min-width: 20px;
        }
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {
            background: #2a2a2a;
            width: 0;
        }

        /* Tooltips */
        QToolTip {
            background-color: #444444;
            color: #f0f0f0;
            border: 1px solid #666666;
        }
    )";

    qApp->setStyleSheet(css);
    qDebug() << "[Theme][UI] Global dark stylesheet applied";
}

