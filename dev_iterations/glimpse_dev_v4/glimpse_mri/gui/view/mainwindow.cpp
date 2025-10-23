// view/mainwindow.cpp
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
#include <QWheelEvent>
#include <QDebug>

#include <algorithm>
#include <chrono>

#include <opencv2/imgproc.hpp>

using namespace std::chrono_literals;

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent)
{
    // Central area with an image label + slice slider
    auto* central = new QWidget(this);
    auto* vlay    = new QVBoxLayout(central);
    vlay->setContentsMargins(8, 8, 8, 8);
    vlay->setSpacing(8);

    m_label = new QLabel(central);
    m_label->setAlignment(Qt::AlignCenter);
    m_label->setText("No image");
    m_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_label->setMinimumSize(320, 240);
    vlay->addWidget(m_label);

    // ---- slice slider (disabled until multi-slice is provided) ----
    m_sliceSlider = new QSlider(Qt::Horizontal, central);
    m_sliceSlider->setMinimum(0);
    m_sliceSlider->setMaximum(0);
    m_sliceSlider->setEnabled(false);
    m_sliceSlider->setVisible(false); // hidden by default
    vlay->addWidget(m_sliceSlider);

    // DnD targets
    setAcceptDrops(true);          // main window
    m_label->setAcceptDrops(true); // central label also receives drops
    showDragHint();                // initial hint

    connect(m_sliceSlider, &QSlider::valueChanged, this, [this](int idx) {
        qDebug() << "[View][Slider] valueChanged ->" << idx;
        if (statusBar()) statusBar()->showMessage(QString("Slice %1").arg(idx + 1), 1500);
        emit sliceChanged(idx);
    });

    m_label->setMouseTracking(true);
    m_label->installEventFilter(this);
    qDebug() << "[View] eventFilter installed on image label for wheel navigation";

    setCentralWidget(central);

    // Metadata dock (hidden by default)
    m_metaDock = new QDockWidget(tr("Metadata"), this);
    m_metaDock->setObjectName("MetaDock");
    m_metaText = new QPlainTextEdit(m_metaDock);
    m_metaText->setReadOnly(true);
    m_metaDock->setWidget(m_metaText);
    addDockWidget(Qt::RightDockWidgetArea, m_metaDock);
    m_metaDock->hide();
    qDebug() << "[View][Meta] Dock created (hidden by default)";

    QStringList m_okExts;  // not const
    m_okExts = { QStringLiteral("dcm"), QStringLiteral("ima"),
                QStringLiteral("h5"),  QStringLiteral("hdf5") };

    // Initial size
    resize(900, 571);
    qDebug() << "[View] Initial size set to" << width() << "x" << height();


}

bool MainWindow::eventFilter(QObject* obj, QEvent* ev)
{
    if (obj == m_label && ev->type() == QEvent::Wheel) {
        auto* we = static_cast<QWheelEvent*>(ev);
        if (!m_sliceSlider || !m_sliceSlider->isEnabled()) {
            qDebug() << "[View][Wheel] slider disabled -> ignore";
            return false; // let others handle it
        }
        const int dir  = (we->angleDelta().y() > 0) ? -1 : +1; // wheel up = previous
        const int cur  = m_sliceSlider->value();
        const int next = std::clamp(cur + dir, m_sliceSlider->minimum(), m_sliceSlider->maximum());
        if (next != cur) {
            qDebug() << "[View][Wheel] slice" << cur << "->" << next;
            m_sliceSlider->setValue(next); // will emit sliceChanged()
        }
        return true; // consumed
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
        showDragHint(); // remind user they can drop files
    }
    enableSliceSlider(0);
}

void MainWindow::setImage(const cv::Mat& img8u)
{
    qDebug() << "[View] setImage ENTER empty=" << img8u.empty()
    << " type=" << img8u.type()
    << " dims=" << img8u.cols << "x" << img8u.rows;

    if (img8u.empty()) {
        qWarning() << "[View] setImage: empty; ignoring";
        return;
    }

    clearDragHint();

    // Keep a local 8-bit single-channel copy in m_img8 (clarity > perf)
    if (img8u.type() != CV_8UC1) {
        qDebug() << "[View] setImage: converting to 8U mono";
        img8u.convertTo(m_img8, CV_8U);
    } else {
        m_img8 = img8u.clone();
    }

    double minv = 0.0, maxv = 0.0;
    cv::minMaxLoc(m_img8, &minv, &maxv);
    qDebug() << "[View] setImage: m_img8 size=" << m_img8.cols << "x" << m_img8.rows
             << " min=" << minv << " max=" << maxv
             << " type=" << m_img8.type();

    m_hasImage = true;

    // Single, guarded repaint
    refreshPixmap();

    qDebug() << "[View] setImage EXIT";
}

void MainWindow::refreshPixmap()
{
    if (m_refreshing) {
        qDebug() << "[View] refreshPixmap: already refreshing; skip";
        return;
    }
    m_refreshing = true; // ---- guard begin ----

    if (!m_hasImage || m_img8.empty() || !m_label) {
        qDebug() << "[View] refreshPixmap: no image or label; skipping";
        m_refreshing = false;
        return;
    }

    const QSize lbl = m_label->size();
    if (lbl.width() <= 1 || lbl.height() <= 1) {
        qDebug() << "[View] refreshPixmap: label too small (" << lbl.width()
        << "x" << lbl.height() << ") -> skip once";
        m_refreshing = false;
        return;
    }

    // Build display image (optionally overlay slice text)
    const cv::Mat* src = &m_img8;
    cv::Mat shown;
    const bool multi =
        (m_sliceSlider && m_sliceSlider->isEnabled() &&
         (m_sliceSlider->maximum() - m_sliceSlider->minimum() + 1) >= 2);

    if (multi) {
        shown = m_img8.clone();   // keep m_img8 pristine
        drawSliceOverlay(shown);  // overlay "Slice i/N"
        src = &shown;
    }

    // QImage that owns its pixels (copy())
    QImage q(src->data, src->cols, src->rows,
             static_cast<int>(src->step), QImage::Format_Grayscale8);
    QImage qc = q.copy();

    // Fast single scaling; AR preserved; avoids heavy smoothing work
    const Qt::AspectRatioMode ar = Qt::KeepAspectRatio;
    const Qt::TransformationMode mode = Qt::FastTransformation;
    const QImage scaled = qc.scaled(lbl, ar, mode);

    QPixmap pm = QPixmap::fromImage(scaled);
    m_label->setPixmap(pm);
    m_label->update();

    qDebug() << "[View] refreshPixmap EXIT  src="
             << src->cols << "x" << src->rows
             << " label=" << lbl.width() << "x" << lbl.height()
             << " pixmap=" << pm.width() << "x" << pm.height();

    m_refreshing = false; // ---- guard end ----
}

void MainWindow::contextMenuEvent(QContextMenuEvent* ev)
{
    QMenu menu(this);

    QAction* actSavePNG    = menu.addAction("Save as PNG...");
    QAction* actSaveDICOM  = menu.addAction("Save as DICOM...");
    menu.addSeparator();
    QAction* actToggleMeta = menu.addAction("Toggle Metadata Dock");

    // Start Over (returns to idle + shows explicit drag hint)
    QAction* actStartOver  = menu.addAction("Start Over (drag a DICOM/HDF5)");

    QAction* chosen = menu.exec(ev->globalPos());
    if (!chosen) return;

    if (chosen == actSavePNG) {
        onSavePNG();
    } else if (chosen == actSaveDICOM) {
        onSaveDICOM();
    } else if (chosen == actToggleMeta) {
        if (m_metaDock) m_metaDock->setVisible(!m_metaDock->isVisible());
    } else if (chosen == actStartOver) {
        qDebug() << "[UI] Start Over selected: clearing to idle with drag hint";
        beginNewImageCycle(); // already shows the drag hint
        if (statusBar())
            statusBar()->showMessage("Reset — drag a DICOM folder or .h5/.dcm onto the window.", 3000);
        emit startOverRequested();     // ask controller to clear its state too
    }
}

void MainWindow::resizeEvent(QResizeEvent* ev)
{
    QMainWindow::resizeEvent(ev);
    qDebug() << "[View] resizeEvent -> refreshPixmap()";
    refreshPixmap();
}

void MainWindow::onSavePNG()
{
    if (!m_hasImage || m_img8.empty()) {
        qWarning() << "[UI] onSavePNG: no image to save";
        return;
    }

    QString pics = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    QString suggested = (pics.isEmpty() ? QDir::homePath() : pics) + "/Glimpse/image.png";
    qDebug() << "[UI] PicturesLocation =" << pics << "| suggested =" << suggested;

    QString fn = QFileDialog::getSaveFileName(this, "Save as PNG", suggested, "PNG (*.png)");
    if (fn.isEmpty()) {
        qDebug() << "[UI] onSavePNG: user canceled";
        return;
    }
    if (!fn.endsWith(".png", Qt::CaseInsensitive)) fn += ".png";
    qDebug() << "[UI] requestSavePNG ->" << fn;
    emit requestSavePNG(fn);
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

void MainWindow::beginBusy(const QString& message)
{
    ++m_busyNesting;
    QApplication::setOverrideCursor(Qt::WaitCursor);
    if (statusBar()) statusBar()->showMessage(message);
    qDebug() << "[View] beginBusy:" << message;
}

void MainWindow::endBusy()
{
    if (m_busyNesting > 0) --m_busyNesting;
    if (m_busyNesting == 0) {
        QApplication::restoreOverrideCursor();
        if (statusBar()) statusBar()->clearMessage();
        qDebug() << "[View] endBusy";
    } else {
        qDebug() << "[View] endBusy (nested level =" << m_busyNesting << ")";
    }
}

void MainWindow::enableSliceSlider(int nSlices)
{
    if (!m_sliceSlider) return;

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
        qDebug() << "[View][Slider] disabled (nSlices =" << nSlices << ")";
    }
}

void MainWindow::setSliceIndex(int idx)
{
    if (!m_sliceSlider) return;
    QSignalBlocker block(m_sliceSlider); // avoid emitting sliceChanged
    m_sliceSlider->setValue(idx);
    qDebug() << "[View][Slider] setSliceIndex ->" << idx << "(signals blocked)";
}

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
