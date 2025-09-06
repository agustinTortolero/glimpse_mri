#include "DicomLoader.h"
#include <QDebug>
#include <QImage>

// DCMTK
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>

// Private helpers for DicomLoader.cpp only (file-scope, static linkage)
static bool getStr(DcmDataset* ds, const DcmTagKey& key, QString& out);
static bool getNum(DcmDataset* ds, const DcmTagKey& key, double& out);


static QImage makeGray8From(DicomImage& di, int frameIndex) {
    const auto status = di.getStatus();
    if (status != EIS_Normal) {
        qWarning() << "[Dicom] DicomImage not normal, status =" << (int)status;
        return {};
    }
    const int w = (int)di.getWidth();
    const int h = (int)di.getHeight();
    qInfo() << "[Dicom] Image size:" << w << "x" << h
            << "depth:" << di.getDepth()
            << "reqFrame:" << frameIndex;

    // Ask DCMTK to scale/window to 8-bit for the requested frame
    auto* buf = new Uint8[w * h];
    if (!di.getOutputData(buf, w * h, 8 /*bits*/, frameIndex /*frame*/)) {
        qWarning() << "[Dicom] getOutputData(8-bit) failed for frame" << frameIndex;
        delete[] buf;
        return {};
    }
    QImage img(w, h, QImage::Format_Grayscale8);
    memcpy(img.bits(), buf, size_t(w) * h);
    delete[] buf;
    return img;
}

static bool loadDicomImage(const QString& path, DcmFileFormat& dfile, std::unique_ptr<DicomImage>& outDi) {
    qInfo() << "[Dicom] Loading:" << path;
    OFCondition cond = dfile.loadFile(path.toStdWString().c_str());
    if (cond.bad()) {
        qWarning() << "[Dicom] loadFile failed:" << cond.text();
        return false;
    }
    // 0,0 tells DCMTK it may manage all frames internally (we will request frames explicitly)
    outDi.reset(new DicomImage(dfile.getDataset(), EXS_Unknown, CIF_AcrNemaCompatibility, 0 /*firstFrame*/, 0 /*frames*/));
    if (outDi->getStatus() != EIS_Normal) {
        qWarning() << "[Dicom] DicomImage error:" << (int)outDi->getStatus();
        return false;
    }
    qInfo() << "[Dicom] PhotometricInterpretation =" << (int)outDi->getPhotometricInterpretation()
            << "Frames =" << outDi->getFrameCount();
    return true;
}

int DicomLoader::countFrames(const QString& path) {
    DcmFileFormat dfile;
    std::unique_ptr<DicomImage> di;
    if (!loadDicomImage(path, dfile, di)) return 0;
    const int frames = (int)di->getFrameCount();
    qInfo() << "[Dicom] countFrames =" << frames;
    return frames;
}

QImage DicomLoader::loadFrame(const QString& path, int frameIndex) {
    DcmFileFormat dfile;
    std::unique_ptr<DicomImage> di;
    if (!loadDicomImage(path, dfile, di)) return {};

    const int frames = (int)di->getFrameCount();
    if (frames <= 0) {
        qWarning() << "[Dicom] No frames present";
        return {};
    }
    if (frameIndex < 0 || frameIndex >= frames) {
        qWarning() << "[Dicom] frameIndex out of range:" << frameIndex << "valid=[0," << (frames-1) << "]";
        return {};
    }

    const auto pi = di->getPhotometricInterpretation();
    if (pi == EPI_Monochrome1 || pi == EPI_Monochrome2) {
        return makeGray8From(*di, frameIndex);
    }

    // Fallback: 24-bit RGB for the requested frame
    const int w = (int)di->getWidth();
    const int h = (int)di->getHeight();
    auto* buf = new Uint8[w * h * 3];
    if (!di->getOutputData(buf, w * h * 3, 8, frameIndex)) {
        qWarning() << "[Dicom] getOutputData(24-bit RGB) failed for frame" << frameIndex;
        delete[] buf;
        return {};
    }
    QImage rgb(w, h, QImage::Format_RGB888);
    memcpy(rgb.bits(), buf, size_t(w) * h * 3);
    delete[] buf;
    return rgb;
}

// Keep your original convenience loader for frame 0
QImage DicomLoader::load(const QString& path) {
    return loadFrame(path, 0);
}

// Small helpers for safely fetching strings and numbers from DICOM
static bool getStr(DcmDataset* ds, const DcmTagKey& key, QString& out) {
    OFString v;
    if (!ds) return false;
    if (ds->findAndGetOFString(key, v).good() && !v.empty()) {
        out = QString::fromLatin1(v.c_str());
        return true;
    }
    return false;
}

static bool getNum(DcmDataset* ds, const DcmTagKey& key, double& out) {
    Float64 f = 0.0;
    if (!ds) return false;
    if (ds->findAndGetFloat64(key, f).good()) {
        out = f;
        return true;
    }
    return false;
}


QString DicomLoader::info(const QString& path) {
    qInfo() << "[Dicom] Reading info for:" << path;

    DcmFileFormat dfile;
    OFCondition cond = dfile.loadFile(path.toStdWString().c_str());
    if (cond.bad()) {
        qWarning() << "[Dicom] loadFile failed in info():" << cond.text();
        return "Failed to load DICOM.\n";
    }
    DcmDataset* ds = dfile.getDataset();

    QStringList lines;
    auto add = [&](const char* key, const QString& val) {
        if (!val.isEmpty()) lines << QString("%1: %2").arg(key, val);
    };
    auto addf = [&](const char* key, double v, const char* unit = nullptr) {
        lines << QString("%1: %2%3").arg(key).arg(v, 0, 'g', 12).arg(unit ? unit : "");
    };

    // Basic Patient / Study / Series
    QString patientName, patientID, patientSex, patientBirth;
    getStr(ds, DCM_PatientName, patientName);
    getStr(ds, DCM_PatientID, patientID);
    getStr(ds, DCM_PatientSex, patientSex);
    getStr(ds, DCM_PatientBirthDate, patientBirth);

    QString studyDate, studyTime, studyDesc, accession;
    getStr(ds, DCM_StudyDate, studyDate);
    getStr(ds, DCM_StudyTime, studyTime);
    getStr(ds, DCM_StudyDescription, studyDesc);
    getStr(ds, DCM_AccessionNumber, accession);

    QString seriesDesc, modality, bodyPart, protocol, sequenceName;
    getStr(ds, DCM_SeriesDescription, seriesDesc);
    getStr(ds, DCM_Modality, modality);
    getStr(ds, DCM_BodyPartExamined, bodyPart);
    getStr(ds, DCM_ProtocolName, protocol);
    getStr(ds, DCM_SequenceName, sequenceName);

    // UIDs
    QString studyUID, seriesUID, sopUID;
    getStr(ds, DCM_StudyInstanceUID, studyUID);
    getStr(ds, DCM_SeriesInstanceUID, seriesUID);
    getStr(ds, DCM_SOPInstanceUID, sopUID);

    // Image geometry
    QString rowsStr, colsStr, photometric;
    getStr(ds, DCM_Rows, rowsStr);
    getStr(ds, DCM_Columns, colsStr);
    getStr(ds, DCM_PhotometricInterpretation, photometric);

    QString pixelSpacing, sliceThickness, spacingBetweenSlices, imageOrientation, imagePosition;
    getStr(ds, DCM_PixelSpacing, pixelSpacing); // format: "rowSpacing\colSpacing"
    getStr(ds, DCM_SliceThickness, sliceThickness);
    getStr(ds, DCM_SpacingBetweenSlices, spacingBetweenSlices);
    getStr(ds, DCM_ImageOrientationPatient, imageOrientation);
    getStr(ds, DCM_ImagePositionPatient, imagePosition);

    // Windowing / bits
    QString winCenter, winWidth, bitsStored, bitsAllocated, highBit, samplesPerPixel;
    getStr(ds, DCM_WindowCenter,  winCenter);
    getStr(ds, DCM_WindowWidth,   winWidth);
    getStr(ds, DCM_BitsStored,    bitsStored);
    getStr(ds, DCM_BitsAllocated, bitsAllocated);
    getStr(ds, DCM_HighBit,       highBit);
    getStr(ds, DCM_SamplesPerPixel, samplesPerPixel);

    // Frame count
    QString numberOfFrames;
    getStr(ds, DCM_NumberOfFrames, numberOfFrames);

    // Manufacturer info
    QString manufacturer, modelName, stationName, softwareVersions;
    getStr(ds, DCM_Manufacturer, manufacturer);
    getStr(ds, DCM_ManufacturerModelName, modelName);
    getStr(ds, DCM_StationName, stationName);
    getStr(ds, DCM_SoftwareVersions, softwareVersions);

    // Acquisition timing
    QString acquisitionDate, acquisitionTime, repetitionTime, echoTime, inversionTime, flipAngle;
    getStr(ds, DCM_AcquisitionDate, acquisitionDate);
    getStr(ds, DCM_AcquisitionTime, acquisitionTime);
    getStr(ds, DCM_RepetitionTime, repetitionTime); // ms
    getStr(ds, DCM_EchoTime, echoTime);             // ms
    getStr(ds, DCM_InversionTime, inversionTime);   // ms
    getStr(ds, DCM_FlipAngle, flipAngle);           // deg

    // Build lines
    lines << "== Patient ==";
    add("Name", patientName);
    add("ID", patientID);
    add("Sex", patientSex);
    add("BirthDate", patientBirth);

    lines << "\n== Study ==";
    add("StudyDate", studyDate);
    add("StudyTime", studyTime);
    add("StudyDescription", studyDesc);
    add("AccessionNumber", accession);
    add("StudyInstanceUID", studyUID);

    lines << "\n== Series ==";
    add("SeriesDescription", seriesDesc);
    add("Modality", modality);
    add("BodyPartExamined", bodyPart);
    add("ProtocolName", protocol);
    add("SequenceName", sequenceName);
    add("SeriesInstanceUID", seriesUID);

    lines << "\n== Image ==";
    add("Rows", rowsStr);
    add("Columns", colsStr);
    add("PhotometricInterpretation", photometric);
    add("SamplesPerPixel", samplesPerPixel);
    add("BitsAllocated", bitsAllocated);
    add("BitsStored", bitsStored);
    add("HighBit", highBit);
    add("NumberOfFrames", numberOfFrames);
    add("PixelSpacing (row\\col mm)", pixelSpacing);
    add("SliceThickness (mm)", sliceThickness);
    add("SpacingBetweenSlices (mm)", spacingBetweenSlices);
    add("ImageOrientationPatient", imageOrientation);
    add("ImagePositionPatient", imagePosition);
    add("WindowCenter", winCenter);
    add("WindowWidth", winWidth);
    add("SOPInstanceUID", sopUID);

    lines << "\n== Acquisition ==";
    add("AcquisitionDate", acquisitionDate);
    add("AcquisitionTime", acquisitionTime);
    add("RepetitionTime (ms)", repetitionTime);
    add("EchoTime (ms)", echoTime);
    add("InversionTime (ms)", inversionTime);
    add("FlipAngle (deg)", flipAngle);

    lines << "\n== System ==";
    add("Manufacturer", manufacturer);
    add("ModelName", modelName);
    add("StationName", stationName);
    add("SoftwareVersions", softwareVersions);

    const QString out = lines.join('\n');
    qInfo() << "[Dicom] Info assembled, lines =" << lines.size();
    return out;
}
