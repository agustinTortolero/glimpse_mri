#include "mainwindow.hpp"

#include <QStatusBar>
#include <QContextMenuEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QDir>
#include <QImage>
#include <QPixmap>
#include <QGuiApplication>
#include <QScreen>
#include <QFile>
#include <QTemporaryFile>
#include <QDateTime>
#include <QRandomGenerator>
#include <QImageWriter>
#include <QStandardPaths>
#include <QVBoxLayout>
#include <QTimer>
#include <chrono>
#include <algorithm>
#include <iostream>

#include <QApplication>
#include <QStatusBar>


using namespace std::chrono_literals;

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent)
{
    // Central area with an image label
    auto* central = new QWidget(this);
    auto* vlay    = new QVBoxLayout(central);
    vlay->setContentsMargins(8, 8, 8, 8);

    m_label = new QLabel(central);
    m_label->setAlignment(Qt::AlignCenter);
    m_label->setText("No image");
    vlay->addWidget(m_label);

    setCentralWidget(central);

    // Metadata dock (hidden by default)
    m_metaDock = new QDockWidget(tr("Metadata"), this);
    m_metaDock->setObjectName("MetaDock");
    m_metaText = new QPlainTextEdit(m_metaDock);
    m_metaText->setReadOnly(true);
    m_metaDock->setWidget(m_metaText);
    addDockWidget(Qt::RightDockWidgetArea, m_metaDock);
    m_metaDock->hide();
    std::cerr << "[DBG][View][Meta] Dock created (hidden by default)\n";

    // Initial size
    resize(900, 571);
    std::cerr << "[DBG][View] Initial size set to " << width() << "x" << height() << "\n";
}

void MainWindow::setMetadata(const QStringList& lines)
{
    if (!m_metaText) return;
    m_metaText->setPlainText(lines.join('\n'));
    std::cerr << "[DBG][View][Meta] setMetadata with " << lines.size() << " line(s)\n";
}

void MainWindow::appendMetadataLine(const QString& line)
{
    if (!m_metaText) return;
    m_metaText->appendPlainText(line);
    std::cerr << "[DBG][View][Meta] append: " << line.toStdString() << "\n";
}

void MainWindow::beginNewImageCycle()
{
    std::cerr << "[DBG][View] beginNewImageCycle(): clear image state\n";
    m_hasImage = false;
    m_img8.release();
    if (m_label) m_label->clear();
}

void MainWindow::setImage(const cv::Mat& img8u)
{
    if (img8u.empty()) {
        std::cerr << "[WRN][View] setImage: empty; ignoring\n";
        return;
    }

    if (img8u.type() != CV_8UC1) {
        std::cerr << "[DBG][View] setImage: converting to 8U mono\n";
        img8u.convertTo(m_img8, CV_8U);
    } else {
        m_img8 = img8u.clone();
    }

    double minv = 0.0, maxv = 0.0;
    cv::minMaxLoc(m_img8, &minv, &maxv);
    std::cerr << "[DBG][View] setImage: m_img8 size=" << m_img8.cols << "x" << m_img8.rows
              << " min=" << minv << " max=" << maxv
              << " type=" << m_img8.type() << "\n";

    m_hasImage = true;

    const QString ts = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss");
    appendMetadataLine(
        QString("[View] %1 Image %2x%3, min=%4 max=%5")
            .arg(ts)
            .arg(m_img8.cols)
            .arg(m_img8.rows)
            .arg(minv, 0, 'f', 3)
            .arg(maxv, 0, 'f', 3)
        );

    refreshPixmap();
    std::cerr << "[DBG][View] Image updated in QLabel.\n";
}

void MainWindow::refreshPixmap()
{
    // 1) If we don’t have an image yet, skip quietly.
    if (!m_hasImage || m_img8.empty()) {
        std::cerr << "[DBG][View] refreshPixmap: no image yet; skipping paint\n";
        return;
    }

    // 2) Avoid painting during early layout (kills tiny 30x30 draws).
    const QSize lbl = m_label->size();
    if (lbl.width() < 200 || lbl.height() < 200) {
        std::cerr << "[DBG][View] refreshPixmap: label too small ("
                  << lbl.width() << "x" << lbl.height()
                  << "), deferring...\n";
        QTimer::singleShot(0ms, this, [this]{ refreshPixmap(); });
        return;
    }

    // 3) Convert cv::Mat → QImage (copy to own memory) → QPixmap → scale → set
    QImage q(m_img8.data, m_img8.cols, m_img8.rows,
             static_cast<int>(m_img8.step), QImage::Format_Grayscale8);
    QImage qc = q.copy(); // own the pixels for safety
    QPixmap pm = QPixmap::fromImage(qc).scaled(
        lbl, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    m_label->setPixmap(pm);

    std::cerr << "[DBG][View] refreshPixmap: img "
              << m_img8.cols << "x" << m_img8.rows
              << " -> label " << lbl.width() << "x" << lbl.height()
              << " pixmap " << pm.width() << "x" << pm.height() << "\n";
}

void MainWindow::contextMenuEvent(QContextMenuEvent* ev)
{
    QMenu menu(this);

    QAction* actSavePNG   = menu.addAction("Save as PNG...");
    QAction* actSaveDICOM = menu.addAction("Save as DICOM...");
    menu.addSeparator();
    QAction* actToggleMeta = menu.addAction("Toggle Metadata Dock");

    QAction* chosen = menu.exec(ev->globalPos());
    if (!chosen) return;

    if (chosen == actSavePNG) {
        onSavePNG();
    } else if (chosen == actSaveDICOM) {
        onSaveDICOM();
    } else if (chosen == actToggleMeta) {
        if (m_metaDock) m_metaDock->setVisible(!m_metaDock->isVisible());
    }
}

void MainWindow::resizeEvent(QResizeEvent* ev)
{
    QMainWindow::resizeEvent(ev);
    refreshPixmap();
}

void MainWindow::onSavePNG()
{
    if (!m_hasImage || m_img8.empty()) {
        std::cerr << "[WRN][UI] onSavePNG: no image to save\n";
        return;
    }

    QString pics = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    QString suggested = (pics.isEmpty() ? QDir::homePath() : pics) + "/Glimpse/image.png";
    std::cerr << "[DBG][UI] PicturesLocation = " << pics.toStdString()
              << " | suggested = " << suggested.toStdString() << "\n";

    QString fn = QFileDialog::getSaveFileName(this, "Save as PNG", suggested, "PNG (*.png)");
    if (fn.isEmpty()) {
        std::cerr << "[DBG][UI] onSavePNG: user canceled\n";
        return;
    }
    if (!fn.endsWith(".png", Qt::CaseInsensitive)) fn += ".png";
    std::cerr << "[DBG][UI] requestSavePNG -> " << fn.toStdString() << "\n";
    emit requestSavePNG(fn);
}

void MainWindow::onSaveDICOM()
{
    if (!m_hasImage || m_img8.empty()) {
        std::cerr << "[WRN][UI] onSaveDICOM: no image to save\n";
        return;
    }

    QString pics = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    QString suggested = (pics.isEmpty() ? QDir::homePath() : pics) + "/Glimpse/image.dcm";
    std::cerr << "[DBG][UI] PicturesLocation = " << pics.toStdString()
              << " | suggested = " << suggested.toStdString() << "\n";

    QString fn = QFileDialog::getSaveFileName(this, "Save as DICOM", suggested, "DICOM (*.dcm)");
    if (fn.isEmpty()) {
        std::cerr << "[DBG][UI] onSaveDICOM: user canceled\n";
        return;
    }
    if (!fn.endsWith(".dcm", Qt::CaseInsensitive)) fn += ".dcm";
    std::cerr << "[DBG][UI] requestSaveDICOM -> " << fn.toStdString() << "\n";
    emit requestSaveDICOM(fn);
}


void MainWindow::beginBusy(const QString& message)
{
    ++m_busyNesting;
    QApplication::setOverrideCursor(Qt::WaitCursor);
    statusBar()->showMessage(message);
    std::cerr << "[DBG][View] beginBusy: " << message.toStdString() << "\n";
}

void MainWindow::endBusy()
{
    if (m_busyNesting > 0) --m_busyNesting;
    if (m_busyNesting == 0) {
        QApplication::restoreOverrideCursor();
        statusBar()->clearMessage();
        std::cerr << "[DBG][View] endBusy\n";
    } else {
        std::cerr << "[DBG][View] endBusy (nested level=" << m_busyNesting << ")\n";
    }
}
