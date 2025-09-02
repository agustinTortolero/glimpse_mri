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


#include <iostream>
#include <algorithm>               // std::min
#include <opencv2/imgcodecs.hpp>   // cv::imwrite

// DCMTK
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcuid.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmdata/dcistrmz.h>
#include <dcmtk/dcmdata/dcmetinf.h>

namespace {
static bool write_dicom_sc_gray8(const cv::Mat& img8, const std::string& path)
{
    try {
        const int rows = img8.rows, cols = img8.cols;
        DcmFileFormat fileformat;
        DcmDataset* ds = fileformat.getDataset();

        ds->putAndInsertString(DCM_SOPClassUID, UID_SecondaryCaptureImageStorage);
        char uid[100]; dcmGenerateUniqueIdentifier(uid, SITE_INSTANCE_UID_ROOT);
        ds->putAndInsertString(DCM_SOPInstanceUID, uid);
        ds->putAndInsertString(DCM_PatientName, "Anon^Patient");
        ds->putAndInsertString(DCM_PatientID, "0000");

        ds->putAndInsertUint16(DCM_SamplesPerPixel, 1);
        ds->putAndInsertString(DCM_PhotometricInterpretation, "MONOCHROME2");
        ds->putAndInsertUint16(DCM_Rows, (Uint16)rows);
        ds->putAndInsertUint16(DCM_Columns, (Uint16)cols);
        ds->putAndInsertUint16(DCM_BitsAllocated, 8);
        ds->putAndInsertUint16(DCM_BitsStored, 8);
        ds->putAndInsertUint16(DCM_HighBit, 7);
        ds->putAndInsertUint16(DCM_PixelRepresentation, 0);
        ds->putAndInsertString(DCM_PixelSpacing, "1\\1");

        const size_t nbytes = (size_t)rows * cols;
        ds->putAndInsertUint8Array(DCM_PixelData, (Uint8*)img8.data, nbytes);

        OFCondition status = fileformat.saveFile(path.c_str(), EXS_LittleEndianExplicit);
        if (status.bad()) {
            std::cerr << "[ERR][DICOM] saveFile failed: " << status.text() << "\n";
            return false;
        }
        std::cerr << "[DBG][DICOM] Saved: " << path << "\n";
        return true;
    } catch (...) {
        std::cerr << "[ERR][DICOM] Exception while writing.\n";
        return false;
    }
}
} // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent)
{
    m_label = new QLabel(this);
    m_label->setAlignment(Qt::AlignCenter);
    setCentralWidget(m_label);
    statusBar()->showMessage("Ready");
    resize(900, 700);

    auto s = QGuiApplication::primaryScreen()->availableGeometry();
    const int wTarget = std::min<int>(width(),  static_cast<int>(s.width()  * 0.7));
    const int hTarget = std::min<int>(height(), static_cast<int>(s.height() * 0.7));
    resize(wTarget, hTarget);
    std::cerr << "[DBG][View] Initial size set to " << wTarget << "x" << hTarget << "\n";
}

void MainWindow::setImage(const cv::Mat& img8u)
{
    if (img8u.empty()) {
        std::cerr << "[ERR][View] setImage: empty\n";
        return;
    }

    // ensure 8-bit mono
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
    QImage qc = q.copy(); // detach

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
    QAction* sel = menu.exec(ev->globalPos());
    if (sel == a1) onSavePNG();
    else if (sel == a2) onSaveDICOM();
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




static bool writeDicomMono8(const cv::Mat& img8, const QString& dstPath)
{
    if (img8.empty() || img8.type() != CV_8UC1) {
        std::cerr << "[ERR][DICOM] writeDicomMono8: invalid image (need CV_8UC1)\n";
        return false;
    }

    // Ensure destination directory exists (Qt is Unicode-friendly)
    const QFileInfo dstInfo(dstPath);
    const QString dstDir = dstInfo.absolutePath();
    if (!QDir().exists(dstDir)) {
        std::cerr << "[DBG][DICOM] mkpath: " << dstDir.toStdString() << "\n";
        if (!QDir().mkpath(dstDir)) {
            std::cerr << "[ERR][DICOM] mkpath failed for: " << dstDir.toStdString() << "\n";
            return false;
        }
    }

    // ----- Build a minimal Secondary Capture DICOM (same as before) -----
    DcmFileFormat ff;
    DcmDataset* ds = ff.getDataset();

    char studyUID[100], seriesUID[100], sopUID[100];
    dcmGenerateUniqueIdentifier(studyUID,  SITE_STUDY_UID_ROOT);
    dcmGenerateUniqueIdentifier(seriesUID, SITE_SERIES_UID_ROOT);
    dcmGenerateUniqueIdentifier(sopUID,    SITE_INSTANCE_UID_ROOT);

    ds->putAndInsertString(DCM_SOPClassUID, UID_SecondaryCaptureImageStorage);
    ds->putAndInsertString(DCM_SOPInstanceUID, sopUID);
    ds->putAndInsertString(DCM_SpecificCharacterSet, "ISO_IR 100");
    ds->putAndInsertString(DCM_PatientName, "Anon^Patient");
    ds->putAndInsertString(DCM_PatientID,   "000000");
    ds->putAndInsertString(DCM_StudyInstanceUID,  studyUID);
    ds->putAndInsertString(DCM_SeriesInstanceUID, seriesUID);
    ds->putAndInsertString(DCM_Modality, "OT");

    const Uint16 rows = static_cast<Uint16>(img8.rows);
    const Uint16 cols = static_cast<Uint16>(img8.cols);
    ds->putAndInsertUint16(DCM_Rows, rows);
    ds->putAndInsertUint16(DCM_Columns, cols);
    ds->putAndInsertString(DCM_PhotometricInterpretation, "MONOCHROME2");
    ds->putAndInsertUint16(DCM_SamplesPerPixel, 1);
    ds->putAndInsertUint16(DCM_BitsAllocated, 8);
    ds->putAndInsertUint16(DCM_BitsStored, 8);
    ds->putAndInsertUint16(DCM_HighBit, 7);
    ds->putAndInsertUint16(DCM_PixelRepresentation, 0);

    const size_t nbytes = static_cast<size_t>(rows) * cols;
    OFCondition st = ds->putAndInsertUint8Array(DCM_PixelData,
                                                img8.data,
                                                static_cast<unsigned long>(nbytes));
    if (st.bad()) {
        std::cerr << "[ERR][DICOM] putAndInsertUint8Array failed: " << st.text() << "\n";
        return false;
    }

    DcmMetaInfo* meta = ff.getMetaInfo();
    meta->putAndInsertString(DCM_MediaStorageSOPClassUID,    UID_SecondaryCaptureImageStorage);
    meta->putAndInsertString(DCM_MediaStorageSOPInstanceUID, sopUID);
    meta->putAndInsertString(DCM_TransferSyntaxUID,          UID_LittleEndianExplicitTransferSyntax);

    // ----- Save to a TEMP ASCII-safe file first (DCMTK char* path) -----
    const QString tmpDir = QDir::tempPath() + "/glimpse_dcm_tmp";
    QDir().mkpath(tmpDir);

    // Use QTemporaryFile to ensure an ASCII-ish path; keep file after close.
    QTemporaryFile tf(tmpDir + "/glimpse_XXXXXX.dcm");
    tf.setAutoRemove(false);
    if (!tf.open()) {
        std::cerr << "[ERR][DICOM] QTemporaryFile open failed in " << tmpDir.toStdString() << "\n";
        return false;
    }
    const QString tmpPath = tf.fileName(); // e.g., C:/Users/.../Temp/glimpse_ab12cd.dcm
    tf.close();

    std::cerr << "[DBG][DICOM] Saving (via DCMTK) to temp: " << tmpPath.toStdString() << "\n";
    // Use narrow path for DCMTK (temp path should be ASCII on Windows)
    const OFCondition saveSt = ff.saveFile(tmpPath.toLocal8Bit().constData(), EXS_LittleEndianExplicit);
    if (saveSt.bad()) {
        std::cerr << "[ERR][DICOM] saveFile(temp) failed: " << saveSt.text()
        << " path=" << tmpPath.toStdString() << "\n";
        QFile::remove(tmpPath);
        return false;
    }

    // ----- Move temp -> destination using Qt (Unicode-capable) -----
    const QString dstNative = QDir::toNativeSeparators(dstPath);
    if (QFile::exists(dstNative)) {
        std::cerr << "[DBG][DICOM] Destination exists, removing: " << dstNative.toStdString() << "\n";
        if (!QFile::remove(dstNative)) {
            std::cerr << "[ERR][DICOM] Failed to remove existing: " << dstNative.toStdString() << "\n";
            QFile::remove(tmpPath);
            return false;
        }
    }

    if (QFile::rename(tmpPath, dstNative)) {
        std::cerr << "[DBG][DICOM] Saved (moved) to: " << dstNative.toStdString() << "\n";
        return true;
    } else {
        std::cerr << "[ERR][DICOM] rename failed, trying copy: "
                  << tmpPath.toStdString() << " -> " << dstNative.toStdString() << "\n";
        if (QFile::copy(tmpPath, dstNative)) {
            QFile::remove(tmpPath);
            std::cerr << "[DBG][DICOM] Saved (copied) to: " << dstNative.toStdString() << "\n";
            return true;
        } else {
            std::cerr << "[ERR][DICOM] copy failed to: " << dstNative.toStdString() << "\n";
            QFile::remove(tmpPath);
            return false;
        }
    }
}


static bool writeDicomMono8(const cv::Mat& img8, const QString& qpath); // forward if needed

void MainWindow::onSaveDICOM()
{
    if (m_img8.empty()) {
        std::cerr << "[ERR][DICOM] No image to save.\n";
        return;
    }

    const QString suggested =
        QDir::homePath() + "/Pictures/Glimpse/" +
        QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + ".dcm";

    const QString path = QFileDialog::getSaveFileName(
        this, "Save as DICOM", suggested, "DICOM (*.dcm)");
    if (path.isEmpty()) return;

    std::cerr << "[DBG][DICOM] Saving to: " << path.toStdString() << "\n";
    if (!writeDicomMono8(m_img8, path)) {
        QMessageBox::warning(this, "DICOM Save",
                             "Failed to save DICOM.\nSee console for details.");
    } else {
        QMessageBox::information(this, "DICOM Save", "DICOM saved successfully.");
    }
}
