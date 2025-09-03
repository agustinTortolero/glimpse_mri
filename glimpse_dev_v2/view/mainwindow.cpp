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
#include <QMenu>              // NEW
#include <QTextOption>        // NEW

#include <iostream>
#include <algorithm>
#include <opencv2/imgcodecs.hpp>
// --- DCMTK includes (FIX: were missing) ---
#include <dcmtk/dcmdata/dctk.h>    // DcmFileFormat, DcmDataset, DcmMetaInfo, OFCondition, tags, UIDs
#include <dcmtk/dcmdata/dcuid.h>   // dcmGenerateUniqueIdentifier, UID_* constants
#include <dcmtk/dcmdata/dcostrmb.h>

// DCMTK includes … (unchanged)

// ... your static write_dicom_sc_gray8(...) and namespace {} remain unchanged ...

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent)
{
    m_label = new QLabel(this);
    m_label->setAlignment(Qt::AlignCenter);
    setCentralWidget(m_label);
    statusBar()->showMessage("Ready");
    resize(900, 700);

    // --- NEW: metadata dock ---
    m_metaDock = new QDockWidget(tr("Metadata"), this);
    m_metaDock->setAllowedAreas(Qt::RightDockWidgetArea | Qt::LeftDockWidgetArea);
    m_metaDock->setFeatures(QDockWidget::DockWidgetMovable |
                            QDockWidget::DockWidgetFloatable |
                            QDockWidget::DockWidgetClosable);
    m_metaText = new QPlainTextEdit(m_metaDock);
    m_metaText->setReadOnly(true);
    m_metaText->setWordWrapMode(QTextOption::NoWrap);
    m_metaDock->setWidget(m_metaText);
    addDockWidget(Qt::RightDockWidgetArea, m_metaDock);

    // Start hidden; user can toggle from context menu
    m_metaDock->hide();
    std::cerr << "[DBG][View][Meta] Dock created (hidden by default)\n";

    // Keep your nice initial sizing
    auto s = QGuiApplication::primaryScreen()->availableGeometry();
    const int wTarget = std::min<int>(width(),  static_cast<int>(s.width()  * 0.7));
    const int hTarget = std::min<int>(height(), static_cast<int>(s.height() * 0.7));
    resize(wTarget, hTarget);
    std::cerr << "[DBG][View] Initial size set to " << wTarget << "x" << hTarget << "\n";
}

void MainWindow::setMetadata(const QStringList& lines)
{
    m_metaText->setPlainText(lines.join("\n"));
    std::cerr << "[DBG][View][Meta] setMetadata with " << lines.size() << " line(s)\n";
}

void MainWindow::setMetadataText(const QString& txt)
{
    m_metaText->setPlainText(txt);
    std::cerr << "[DBG][View][Meta] setMetadataText length=" << txt.size() << "\n";
}

void MainWindow::appendMetadataLine(const QString& line)
{
    m_metaText->appendPlainText(line);
    std::cerr << "[DBG][View][Meta] append: " << line.toStdString() << "\n";
}

void MainWindow::toggleMetadataPanel()
{
    const bool vis = m_metaDock->isVisible();
    m_metaDock->setVisible(!vis);
    statusBar()->showMessage(vis ? "Metadata hidden" : "Metadata shown", 1200);
    std::cerr << "[DBG][View][Meta] toggle -> " << (!vis ? "show" : "hide") << "\n";
}

void MainWindow::setImage(const cv::Mat& img8u)
{
    if (img8u.empty()) {
        std::cerr << "[ERR][View] setImage: empty\n";
        return;
    }

    if (img8u.type() != CV_8UC1) {
        std::cerr << "[DBG][View] setImage: converting to 8U mono\n";
        img8u.convertTo(m_img8, CV_8U);
    } else {
        m_img8 = img8u.clone();
    }

    double minv = 0, maxv = 0;
    cv::minMaxLoc(m_img8, &minv, &maxv);
    std::cerr << "[DBG][View] setImage: m_img8 size=" << m_img8.cols << "x" << m_img8.rows
              << " min=" << minv << " max=" << maxv
              << " type=" << m_img8.type() << "\n";

    // NEW: drop a helpful line when images arrive
    const QString ts = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss");
    appendMetadataLine(QString("[View] %1 Image %2x%3, min=%4 max=%5")
                           .arg(ts).arg(m_img8.cols).arg(m_img8.rows)
                           .arg(minv, 0, 'f', 3).arg(maxv, 0, 'f', 3));

    refreshPixmap();
    std::cerr << "[DBG][View] Image updated in QLabel.\n";
}

void MainWindow::refreshPixmap()
{
    if (m_img8.empty()) {
        std::cerr << "[ERR][View] refreshPixmap: m_img8 is empty\n";
        return;
    }
    QImage q(m_img8.data, m_img8.cols, m_img8.rows,
             static_cast<int>(m_img8.step), QImage::Format_Grayscale8);
    QImage qc = q.copy();

    const int lw = m_label->width();
    const int lh = m_label->height();

    QPixmap pm = QPixmap::fromImage(qc).scaled(
        lw, lh, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    m_label->setPixmap(pm);

    std::cerr << "[DBG][View] refreshPixmap: img "
              << m_img8.cols << "x" << m_img8.rows
              << " -> label " << lw << "x" << lh
              << " pixmap " << pm.width() << "x" << pm.height()
              << "\n";
}

void MainWindow::resizeEvent(QResizeEvent* ev)
{
    QMainWindow::resizeEvent(ev);
    refreshPixmap();
}

void MainWindow::contextMenuEvent(QContextMenuEvent* ev)
{
    QMenu menu(this);
    QAction* a1 = menu.addAction("Save as PNG...");
    QAction* a2 = menu.addAction("Save as DICOM...");
    menu.addSeparator();
    QAction* a3 = menu.addAction(m_metaDock->isVisible() ? "Hide Metadata Panel" : "Show Metadata Panel");

    QAction* sel = menu.exec(ev->globalPos());
    if (sel == a1) onSavePNG();
    else if (sel == a2) onSaveDICOM();
    else if (sel == a3) toggleMetadataPanel();
}

// === DICOM helpers ===========================================================
static bool write_dicom_sc_gray8(const QString& outPath, const cv::Mat& img8, QString* why)
{
    // Validate
    if (img8.empty()) {
        if (why) *why = "empty image";
        std::cerr << "[ERR][DICOM] write_dicom_sc_gray8: empty image\n";
        return false;
    }
    if (img8.type() != CV_8UC1) {
        if (why) *why = "expected CV_8UC1";
        std::cerr << "[ERR][DICOM] write_dicom_sc_gray8: expected CV_8UC1\n";
        return false;
    }

    // Ensure destination folder exists (Qt is Unicode-safe)
    const QString dstDir = QFileInfo(outPath).absolutePath();
    if (!QDir().mkpath(dstDir)) {
        if (why) *why = "Cannot create output directory: " + dstDir;
        std::cerr << "[ERR][DICOM] mkpath failed for: " << dstDir.toStdString() << "\n";
        return false;
    }

    try {
        // --- Build minimal Secondary Capture dataset ---
        DcmFileFormat ff;
        DcmDataset* ds = ff.getDataset();

        char studyUID[128]  = {0};
        char seriesUID[128] = {0};
        char instUID[128]   = {0};
        dcmGenerateUniqueIdentifier(studyUID);
        dcmGenerateUniqueIdentifier(seriesUID);
        dcmGenerateUniqueIdentifier(instUID);

        ds->putAndInsertString(DCM_SOPClassUID,          UID_SecondaryCaptureImageStorage);
        ds->putAndInsertString(DCM_SOPInstanceUID,       instUID);
        ds->putAndInsertString(DCM_SpecificCharacterSet, "ISO_IR 192"); // UTF-8
        ds->putAndInsertString(DCM_PatientName,          "Anon^Patient");
        ds->putAndInsertString(DCM_PatientID,            "0000");
        ds->putAndInsertString(DCM_StudyInstanceUID,     studyUID);
        ds->putAndInsertString(DCM_SeriesInstanceUID,    seriesUID);
        ds->putAndInsertString(DCM_Modality,             "OT");

        const Uint16 rows = static_cast<Uint16>(img8.rows);
        const Uint16 cols = static_cast<Uint16>(img8.cols);
        ds->putAndInsertUint16(DCM_Rows,                    rows);
        ds->putAndInsertUint16(DCM_Columns,                 cols);
        ds->putAndInsertString(DCM_PhotometricInterpretation, "MONOCHROME2");
        ds->putAndInsertUint16(DCM_SamplesPerPixel,         1);
        ds->putAndInsertUint16(DCM_BitsAllocated,           8);
        ds->putAndInsertUint16(DCM_BitsStored,              8);
        ds->putAndInsertUint16(DCM_HighBit,                 7);
        ds->putAndInsertUint16(DCM_PixelRepresentation,     0);

        const Uint32 nbytes = static_cast<Uint32>(img8.total());
        const Uint8* src    = reinterpret_cast<const Uint8*>(img8.data);
        OFCondition st = ds->putAndInsertUint8Array(DCM_PixelData, src, nbytes);
        if (st.bad()) {
            if (why) *why = QString("PixelData insert failed: %1").arg(st.text());
            std::cerr << "[ERR][DICOM] PixelData insert failed: " << st.text() << "\n";
            return false;
        }

        // --- Save to ASCII-safe temp file using DCMTK (narrow path) ---
        const QString tmpBase = QDir::tempPath() + "/glimpse_dcm_tmp";
        QDir().mkpath(tmpBase);
        QTemporaryFile tf(tmpBase + "/glimpse_XXXXXX.dcm");
        tf.setAutoRemove(false);
        if (!tf.open()) {
            if (why) *why = "QTemporaryFile open failed in: " + tmpBase;
            std::cerr << "[ERR][DICOM] QTemporaryFile open failed in " << tmpBase.toStdString() << "\n";
            return false;
        }
        const QString tmpPath = tf.fileName();
        tf.close(); // DCMTK will open/write by its own path handling

        std::cerr << "[DBG][DICOM] Writing SC (temp): "
                  << QDir::toNativeSeparators(tmpPath).toStdString()
                  << " (" << cols << "x" << rows << ", " << nbytes << " bytes)\n";

        // Use local 8-bit for DCMTK (temp path should be ASCII-ish on Windows)
        OFCondition saveSt = ff.saveFile(tmpPath.toLocal8Bit().constData(), EXS_LittleEndianExplicit);
        if (saveSt.bad()) {
            if (why) *why = QString("saveFile(temp) failed: %1").arg(saveSt.text());
            std::cerr << "[ERR][DICOM] saveFile(temp) failed: " << saveSt.text() << "\n";
            QFile::remove(tmpPath);
            return false;
        }

        // --- Move/copy temp -> final Unicode path using Qt ---
        const QString dstNative = QDir::toNativeSeparators(outPath);
        if (QFile::exists(dstNative)) {
            std::cerr << "[DBG][DICOM] Destination exists, removing: " << dstNative.toStdString() << "\n";
            if (!QFile::remove(dstNative)) {
                if (why) *why = "Failed to remove existing destination: " + dstNative;
                std::cerr << "[ERR][DICOM] remove(existing) failed\n";
                QFile::remove(tmpPath);
                return false;
            }
        }

        if (QFile::rename(tmpPath, dstNative)) {
            std::cerr << "[DBG][DICOM] Saved (moved) to: " << dstNative.toStdString() << "\n";
            return true;
        }

        std::cerr << "[WARN][DICOM] rename failed; trying copy\n";
        if (QFile::copy(tmpPath, dstNative)) {
            QFile::remove(tmpPath);
            std::cerr << "[DBG][DICOM] Saved (copied) to: " << dstNative.toStdString() << "\n";
            return true;
        } else {
            if (why) *why = "Final copy failed to: " + dstNative;
            std::cerr << "[ERR][DICOM] copy failed to: " << dstNative.toStdString() << "\n";
            QFile::remove(tmpPath);
            return false;
        }
    }
    catch (const std::exception& ex) {
        if (why) *why = QString("exception: %1").arg(ex.what());
        std::cerr << "[ERR][DICOM] exception: " << ex.what() << "\n";
        return false;
    }
}



void MainWindow::onSavePNG()
{
    if (m_img8.empty()) {
        std::cerr << "[ERR][View] onSavePNG: no image\n";
        return;
    }

    // Suggest a Pictures/Glimpse path (Unicode-friendly)
    QString suggested = QDir::homePath() + "/Pictures/Glimpse/recon.png";
    QString fn = QFileDialog::getSaveFileName(
        this, "Save PNG", suggested, "PNG Images (*.png)");
    if (fn.isEmpty()) return;

    // Ensure .png extension
    if (!fn.endsWith(".png", Qt::CaseInsensitive))
        fn += ".png";

    // Make sure target folder exists
    const QString outDir = QFileInfo(fn).absolutePath();
    if (!QDir().exists(outDir)) {
        std::cerr << "[DBG][PNG] mkpath: " << outDir.toStdString() << "\n";
        if (!QDir().mkpath(outDir)) {
            std::cerr << "[ERR][PNG] mkpath failed for: " << outDir.toStdString() << "\n";
            statusBar()->showMessage("Failed to create target folder", 3000);
            return;
        }
    }

    // Build a QImage from m_img8 (8-bit mono), then detach with copy()
    QImage qi(m_img8.data, m_img8.cols, m_img8.rows,
              static_cast<int>(m_img8.step), QImage::Format_Grayscale8);
    QImage qcopy = qi.copy();

    // Save using Qt (Unicode-safe)
    QImageWriter writer(fn, "png");
    writer.setCompression(9); // 0..9, higher = smaller/slower
    const bool ok = writer.write(qcopy);

    if (!ok) {
        std::cerr << "[ERR][PNG] QImageWriter failed: "
                  << writer.errorString().toStdString() << "\n";
        statusBar()->showMessage("Failed to save PNG", 3000);
    } else {
        std::cerr << "[DBG][PNG] Saved: " << QDir::toNativeSeparators(fn).toStdString() << "\n";
        statusBar()->showMessage("Saved PNG: " + fn, 3000);
    }
}





void MainWindow::onSaveDICOM()
{
    if (m_img8.empty()) {
        std::cerr << "[ERR][UI] onSaveDICOM: no image loaded\n";
        QMessageBox::warning(this, "Save DICOM", "No image loaded.");
        return;
    }

    QString out = QFileDialog::getSaveFileName(this, "Save as DICOM", "image.dcm", "DICOM (*.dcm)");
    if (out.isEmpty()) {
        std::cerr << "[DBG][UI] onSaveDICOM: user canceled\n";
        return;
    }
    if (!out.endsWith(".dcm", Qt::CaseInsensitive))
        out += ".dcm";

    QString why;
    std::cerr << "[DBG][UI] Save DICOM -> " << out.toStdString() << "\n";
    const bool ok = write_dicom_sc_gray8(out, m_img8, &why);
    if (!ok) {
        std::cerr << "[ERR][UI] DICOM save failed: " << why.toStdString() << "\n";
        QMessageBox::critical(this, "DICOM save failed", why);
        return;
    }
    statusBar()->showMessage("Saved DICOM: " + out, 2000);
}
