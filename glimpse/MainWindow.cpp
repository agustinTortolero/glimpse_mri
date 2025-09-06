#include "MainWindow.h"
#include "DicomLoader.h"

#include <QShortcut>
#include <QKeySequence>

#include <QAction>
#include <QContextMenuEvent>
#include <QDebug>
#include <QFileDialog>
#include <QKeyEvent>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPixmap>
#include <QScrollArea>
#include <QScrollBar>
#include <QSlider>
#include <QStatusBar>
#include <QWheelEvent>
#include <algorithm> // std::clamp
#include <cmath>     // std::pow

#include <QDockWidget>
#include <QTextEdit>
#include <QShortcut>
#include <QKeySequence>
#include <QScrollBar>
#include <algorithm>
#include <cmath>

#include <QStandardPaths>
#include <QDir>
#include <QFileInfo>


// ---- Constructor ------------------------------------------------------------

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("Qt + DCMTK DICOM Viewer (Slices + Zoom + Info)");

    // Central image area (unchanged)
    imageLabel_ = new QLabel;
    imageLabel_->setAlignment(Qt::AlignCenter);
    scrollArea_ = new QScrollArea;
    scrollArea_->setWidgetResizable(true);
    scrollArea_->setWidget(imageLabel_);
    setCentralWidget(scrollArea_);

    // File menu (unchanged)
    auto* fileMenu = menuBar()->addMenu("&File");
    auto* actOpen  = fileMenu->addAction("Open &DICOM...");
    connect(actOpen, &QAction::triggered, this, &MainWindow::openDicom);

    // Slice slider (unchanged)
    sliceSlider_ = new QSlider(Qt::Horizontal, this);
    sliceSlider_->setEnabled(false);
    sliceSlider_->setRange(0, 0);
    sliceSlider_->setSingleStep(1);
    statusBar()->addPermanentWidget(sliceSlider_, 1);
    connect(sliceSlider_, &QSlider::valueChanged, this, &MainWindow::onSliceChanged);

    // === NEW: DICOM Details dock ===
    auto* dock = new QDockWidget("DICOM Details", this);
    dock->setObjectName("dicomDetailsDock");
    dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    infoEdit_ = new QTextEdit(dock);
    infoEdit_->setReadOnly(true);
    infoEdit_->setWordWrapMode(QTextOption::NoWrap);
    infoEdit_->setMinimumWidth(360);
    dock->setWidget(infoEdit_);
    addDockWidget(Qt::RightDockWidgetArea, dock);
    qInfo() << "[GUI] Info dock created";

    // Focus & shortcuts (so Ctrl+Arrows always work)
    setFocusPolicy(Qt::StrongFocus);
    if (centralWidget()) centralWidget()->setFocusPolicy(Qt::ClickFocus);

    auto* prevSlice = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_Left),  this);
    auto* nextSlice = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_Right), this);
    connect(prevSlice, &QShortcut::activated, this, [this]{
        if (!sliceSlider_ || !sliceSlider_->isEnabled()) return;
        int v = std::max(sliceSlider_->value() - 1, sliceSlider_->minimum());
        qInfo() << "[GUI] CTRL+Left -> slice" << v;
        sliceSlider_->setValue(v);
    });
    connect(nextSlice, &QShortcut::activated, this, [this]{
        if (!sliceSlider_ || !sliceSlider_->isEnabled()) return;
        int v = std::min(sliceSlider_->value() + 1, sliceSlider_->maximum());
        qInfo() << "[GUI] CTRL+Right -> slice" << v;
        sliceSlider_->setValue(v);
    });

    // Optional: Ctrl+Up/Down zoom
    auto* zoomIn  = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_Up),   this);
    auto* zoomOut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_Down), this);
    connect(zoomIn,  &QShortcut::activated, this, [this]{ qInfo() << "[GUI] CTRL+Up -> zoom in";  zoomBy(1.10, rect().center()); });
    connect(zoomOut, &QShortcut::activated, this, [this]{ qInfo() << "[GUI] CTRL+Down -> zoom out"; zoomBy(1.0/1.10, rect().center()); });

    // Zoom state
    scale_ = 1.0; minScale_ = 0.1; maxScale_ = 8.0;

    qInfo() << "[GUI] MainWindow ready";
}



// ---- Context menu -----------------------------------------------------------

void MainWindow::contextMenuEvent(QContextMenuEvent* ev) {
    QMenu menu(this);
    auto* saveP = menu.addAction("Save as PNG...");
    auto* saveB = menu.addAction("Save as BMP...");

    connect(saveP, &QAction::triggered, this, &MainWindow::savePng);
    connect(saveB, &QAction::triggered, this, &MainWindow::saveBmp);

    qDebug() << "[GUI] Context menu opened";
    menu.exec(ev->globalPos());
}

// ---- File open --------------------------------------------------------------

void MainWindow::openDicom() {
    static QString s_lastDir; // remembers the last folder during this run
    if (s_lastDir.isEmpty()) {
        // Default to Documents (fallback to current working dir)
        s_lastDir = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
        if (s_lastDir.isEmpty()) s_lastDir = QDir::currentPath();
    }

    const QString filter =
        "DICOM (*.dcm *.dicom *.ima);;"
        "All files (*.*)";

    qInfo() << "[GUI] File dialog open. startDir=" << s_lastDir << "filter=" << filter;
    const QString path = QFileDialog::getOpenFileName(
        this,
        "Open DICOM",
        s_lastDir,
        filter,
        /*selectedFilter*/nullptr,
        QFileDialog::DontResolveSymlinks | QFileDialog::ReadOnly
        );

    if (path.isEmpty()) {
        qDebug() << "[GUI] Open canceled";
        return;
    }

    // Update last dir
    QFileInfo fi(path);
    s_lastDir = fi.absolutePath();
    qInfo() << "[GUI] Selected file =" << path;

    if (!fi.exists() || !fi.isFile()) {
        QMessageBox::warning(this, "Open DICOM", "Selected path is not a file.");
        qWarning() << "[GUI] Not a regular file:" << path;
        return;
    }

    openDicomAt(path);  // will handle frames, image display, and status messages
}


void MainWindow::openDicomAt(const QString& path) {
    currentPath_ = path;
    qInfo() << "[GUI] Opening (direct)" << currentPath_;

    // Query number of frames/slices
    frames_ = DicomLoader::countFrames(currentPath_);
    if (frames_ <= 0) {
        // Fallback: try loading as a single-frame image and show it anyway
        qWarning() << "[GUI] No frames reported; trying single-frame load";
        const QImage img = DicomLoader::load(currentPath_);
        if (img.isNull()) {
            QMessageBox::warning(this, "Open DICOM", "Failed to load image.");
            qWarning() << "[GUI] Failed to load image (single-frame fallback)";
            return;
        }
        frame_ = 0;
        setImage(img);
        sliceSlider_->setEnabled(false);
        statusBar()->showMessage("Single-frame DICOM | Zoom 100%");
        return;
    }

    updateInfoPanel(currentPath_);

    // Enable and initialize the slider for [0..frames_-1]
    sliceSlider_->setEnabled(true);
    sliceSlider_->setRange(0, frames_ - 1);

    // Load frame 0
    frame_ = 0;
    const QImage img0 = DicomLoader::loadFrame(currentPath_, frame_);
    if (img0.isNull()) {
        QMessageBox::warning(this, "Open DICOM", "Failed to load first frame.");
        qWarning() << "[GUI] Failed to load frame 0";
        sliceSlider_->setEnabled(false);
        return;
    }
    setImage(img0);
    sliceSlider_->setValue(0);
    statusBar()->showMessage(QString("Slice %1 / %2 | Zoom %3%")
                                 .arg(frame_ + 1).arg(frames_).arg(int(scale_ * 100)));
}

// ---- Slice slider / keyboard nav -------------------------------------------

void MainWindow::onSliceChanged(int v) {
    if (currentPath_.isEmpty()) return;
    if (v == frame_) return;

    qInfo() << "[GUI] Slice slider changed ->" << v;
    frame_ = v;

    const QImage img = DicomLoader::loadFrame(currentPath_, frame_);
    if (img.isNull()) {
        qWarning() << "[GUI] Failed to load frame" << frame_;
        return;
    }
    setImage(img);
    statusBar()->showMessage(QString("Slice %1 / %2 | Zoom %3%")
                                 .arg(frame_ + 1).arg(frames_).arg(int(scale_ * 100)));

    if (infoEdit_) {
        auto t = infoEdit_->toPlainText();
        // Replace/add a last line with current slice
        t = t + QString("\nCurrentSlice: %1 / %2").arg(frame_ + 1).arg(frames_);
        infoEdit_->setPlainText(t);
    }
}
// === DICOM details panel updater ============================================
void MainWindow::updateInfoPanel(const QString& path) {
    if (!infoEdit_) {
        qWarning() << "[GUI] Info panel not created; skipping update";
        return;
    }
    const QString text = DicomLoader::info(path);
    infoEdit_->setPlainText(text);
    qInfo() << "[GUI] Info panel updated (lines =" << (text.count('\n') + 1) << ")";
}


void MainWindow::keyPressEvent(QKeyEvent* ev) {
    // Arrow keys to browse slices (if slider is active)
    if (sliceSlider_ && sliceSlider_->isEnabled()) {
        if (ev->key() == Qt::Key_Up || ev->key() == Qt::Key_Right) {
            const int v = std::min(sliceSlider_->value() + 1, sliceSlider_->maximum());
            if (v != sliceSlider_->value()) sliceSlider_->setValue(v);
            return;
        }
        if (ev->key() == Qt::Key_Down || ev->key() == Qt::Key_Left) {
            const int v = std::max(sliceSlider_->value() - 1, sliceSlider_->minimum());
            if (v != sliceSlider_->value()) sliceSlider_->setValue(v);
            return;
        }
    }
    QMainWindow::keyPressEvent(ev);
}

// ---- Wheel gestures: CTRL+scroll zoom, CTRL+left/right scroll slices -------

void MainWindow::wheelEvent(QWheelEvent* ev) {
    const bool ctrl = (ev->modifiers() & Qt::ControlModifier);
    if (!ctrl) {
        // Pass through normal wheel behavior when CTRL isn't pressed
        QMainWindow::wheelEvent(ev);
        return;
    }

    // Qt wheel: angleDelta is in 1/8 deg units; 15 deg = 120 units per "step"
    const QPoint numDeg = ev->angleDelta() / 8;
    const int stepsY = numDeg.y() / 15; // vertical wheel steps
    const int stepsX = numDeg.x() / 15; // horizontal wheel steps (left/right)

    // CTRL + horizontal scroll -> change slice
    if (stepsX != 0 && sliceSlider_ && sliceSlider_->isEnabled()) {
        const int old = sliceSlider_->value();
        int next = old + stepsX;
        next = std::clamp(next, sliceSlider_->minimum(), sliceSlider_->maximum());
        qInfo() << "[GUI] CTRL+HWheel slice:" << old << "->" << next
                << "(stepX=" << stepsX << ")";
        if (next != old) sliceSlider_->setValue(next);
        ev->accept();
        return;
    }

    // CTRL + vertical scroll -> zoom
    if (stepsY != 0) {
        const double factor = std::pow(1.10, stepsY); // 10% per wheel step
        qInfo() << "[GUI] CTRL+Wheel zoom stepY=" << stepsY << " factor=" << factor;
        zoomBy(factor, ev->position());
        ev->accept();
        return;
    }

    // If neither path took it, fall back
    QMainWindow::wheelEvent(ev);
}

// ---- Save -------------------------------------------------------------------

void MainWindow::savePng() {
    if (current_.isNull()) {
        qWarning() << "[GUI] No image to save (PNG)";
        return;
    }
    const QString out = QFileDialog::getSaveFileName(
        this, "Save PNG", "image.png", "PNG (*.png)");
    if (out.isEmpty()) return;

    const bool ok = current_.save(out, "PNG");
    qInfo() << "[GUI] Save PNG ->" << out << "ok=" << ok;
    if (!ok) QMessageBox::warning(this, "Save PNG", "Failed to save PNG.");
}

void MainWindow::saveBmp() {
    if (current_.isNull()) {
        qWarning() << "[GUI] No image to save (BMP)";
        return;
    }
    const QString out = QFileDialog::getSaveFileName(
        this, "Save BMP", "image.bmp", "BMP (*.bmp)");
    if (out.isEmpty()) return;

    const bool ok = current_.save(out, "BMP");
    qInfo() << "[GUI] Save BMP ->" << out << "ok=" << ok;
    if (!ok) QMessageBox::warning(this, "Save BMP", "Failed to save BMP.");
}

// ---- Image set + scaling ----------------------------------------------------

void MainWindow::setImage(const QImage& img) {
    current_ = img;
    applyScale(); // render with current zoom
    qInfo() << "[GUI] Image set, size =" << current_.size()
            << "scale=" << scale_ << " -> drawn="
            << imageLabel_->pixmap(Qt::ReturnByValue).size();
}

// Scale around mouse position (approximate keep-point behavior)
void MainWindow::zoomBy(double factor, const QPointF& mousePosInViewport) {
    const double oldScale = scale_;
    const double newScale = std::clamp(scale_ * factor, minScale_, maxScale_);
    if (std::abs(newScale - oldScale) < 1e-6) {
        qDebug() << "[GUI] Zoom clamped at" << newScale;
        return;
    }

    // Attempt to keep the mouse position stable by adjusting scrollbars
    // Map mouse position in viewport to content position before scaling
    const QPoint vpPos = mousePosInViewport.toPoint();
    QPoint  hsb(vpPos.x() + scrollArea_->horizontalScrollBar()->value(),
               vpPos.y() + scrollArea_->verticalScrollBar()->value());

    scale_ = newScale;
    applyScale();

    // Compute scale ratio and re-center to keep the same content point under cursor
    const double r = newScale / oldScale;
    scrollArea_->horizontalScrollBar()->setValue(int(hsb.x() * r - vpPos.x()));
    scrollArea_->verticalScrollBar()->setValue(int(hsb.y() * r - vpPos.y()));

    statusBar()->showMessage(QString("Slice %1 / %2 | Zoom %3%")
                                 .arg(frame_ + 1).arg(frames_).arg(int(scale_ * 100)), 1200);
    qInfo() << "[GUI] Zoom:" << oldScale << "->" << newScale
            << "(factor=" << factor << ")";
}

void MainWindow::applyScale() {
    if (current_.isNull()) return;
    const QSize target = (scale_ == 1.0)
                             ? current_.size()
                             : QSize(int(current_.width() * scale_), int(current_.height() * scale_));

    QImage scaled = current_.scaled(
        target, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    imageLabel_->setPixmap(QPixmap::fromImage(scaled));
    imageLabel_->adjustSize();
}
