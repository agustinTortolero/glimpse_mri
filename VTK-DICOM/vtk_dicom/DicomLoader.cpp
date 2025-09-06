#include "DicomLoader.h"

#include <QFileInfo>
#include <QDir>
#include <QDebug>
#include <QStringList>

#include <memory>
#include <cstring>  // std::memcpy

// ================= VTK =================
#include <vtkSmartPointer.h>
#include <vtkNew.h>
#include <vtkImageData.h>
#include <vtkDICOMImageReader.h>
#include <vtkStreamingDemandDrivenPipeline.h>
#include <vtkInformation.h>

// =============== DCMTK =================
// core
#include <dcmtk/config/osconfig.h>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmjpeg/djdecode.h>   // DJDecoderRegistration
#include <dcmtk/dcmjpls/djdecode.h>   // DJLSDecoderRegistration

// DicomImage (handle both include layouts & compilers)
#if defined(__has_include)
#if __has_include(<dcmtk/dcmimage/diregist.h>) && (__has_include(<dcmtk/dcmimage/dicoimg.h>) || __has_include(<dcmtk/dcmimage/dicomimage.h>))
#define HAVE_DCMTK_DICOMIMAGE 1
#include <dcmtk/dcmimage/diregist.h>
#if __has_include(<dcmtk/dcmimage/dicoimg.h>)
#include <dcmtk/dcmimage/dicoimg.h>
#else
#include <dcmtk/dcmimage/dicomimage.h>
#endif
#elif __has_include(<dcmimage/diregist.h>) && (__has_include(<dcmimage/dicoimg.h>) || __has_include(<dcmimage/dicomimage.h>))
#define HAVE_DCMTK_DICOMIMAGE 1
#include <dcmimage/diregist.h>
#if __has_include(<dcmimage/dicoimg.h>)
#include <dcmimage/dicoimg.h>
#else
#include <dcmimage/dicomimage.h>
#endif
#endif
#endif

// ----------------- local helpers -----------------
static QString qExtentStr(const int e[6]) {
    return QString("[%1,%2] x [%3,%4] x [%5,%6]")
    .arg(e[0]).arg(e[1]).arg(e[2]).arg(e[3]).arg(e[4]).arg(e[5]);
}

static double getDoubleOr(DcmItem* ds, const DcmTagKey& tag, double fallback)
{
    if (!ds) return fallback;
    OFString s;
    if (ds->findAndGetOFString(tag, s).good()) {
        bool ok = false;
        const double v = QString::fromLatin1(s.c_str()).toDouble(&ok);
        return ok ? v : fallback;
    }
    return fallback;
}

static int getIntOr(DcmItem* ds, const DcmTagKey& tag, int fallback)
{
    if (!ds) return fallback;
    Sint32 s32 = 0;
    if (ds->findAndGetSint32(tag, s32).good()) return static_cast<int>(s32);
    Uint16 u16 = 0;
    if (ds->findAndGetUint16(tag, u16).good()) return static_cast<int>(u16);
    OFString os;
    if (ds->findAndGetOFString(tag, os).good()) {
        bool ok = false;
        const int v = QString::fromLatin1(os.c_str()).toInt(&ok);
        if (ok) return v;
    }
    return fallback;
}

// -------------------------------------------------
// Public API
// -------------------------------------------------
QString DicomLoader::normalizeDicomPath(const QString& inPath)
{
    QFileInfo fi(inPath);
    if (fi.isFile()) {
        qInfo() << "[DicomLoader] Path is a file, using parent directory for series:"
                << fi.dir().absolutePath();
        return fi.dir().absolutePath();
    }
    return fi.absoluteFilePath();
}

bool DicomLoader::tryVtkReadFile(const QString& filePath,
                                 vtkSmartPointer<vtkImageData>& out,
                                 bool& outIsVolume3D)
{
    qInfo() << "[DicomLoader] tryVtkReadFile =" << QDir::toNativeSeparators(filePath);
    vtkNew<vtkDICOMImageReader> r;
    r->SetFileName(QDir::fromNativeSeparators(filePath).toUtf8().constData());
    r->UpdateInformation();

    int whole[6] = {0,0,0,0,0,0};
    r->GetOutputInformation(0)->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), whole);
    qDebug() << "[VTK] WHOLE_EXTENT (pre-Update) =" << qExtentStr(whole);

    r->Update();
    const unsigned long err = r->GetErrorCode();
    if (err != 0) {
        qWarning() << "[VTK] Reader error code =" << err << "(will try DCMTK fallback)";
        return false;
    }

    vtkImageData* img = r->GetOutput();
    if (!img) return false;

    int ext[6]; img->GetExtent(ext);
    const int zSlices = ext[5] - ext[4] + 1;
    outIsVolume3D = (zSlices > 1);
    out = img;
    return true;
}

bool DicomLoader::tryVtkReadDir(const QString& dirPath,
                                vtkSmartPointer<vtkImageData>& out,
                                bool& outIsVolume3D)
{
    qInfo() << "[DicomLoader] tryVtkReadDir =" << QDir::toNativeSeparators(dirPath);
    vtkNew<vtkDICOMImageReader> r;
    r->SetDirectoryName(QDir::fromNativeSeparators(dirPath).toUtf8().constData());
    r->UpdateInformation();
    r->Update();

    const unsigned long err = r->GetErrorCode();
    if (err != 0) {
        qWarning() << "[VTK] Reader error code =" << err << "(will try DCMTK fallback)";
        return false;
    }

    vtkImageData* img = r->GetOutput();
    if (!img) return false;

    int ext[6]; img->GetExtent(ext);
    const int z = ext[5] - ext[4] + 1;
    outIsVolume3D = (z > 1);
    out = img;
    return true;
}

bool DicomLoader::readSingleFileViaDCMTK(const QString& filePath,
                                         vtkSmartPointer<vtkImageData>& out,
                                         bool& outIsVolume3D)
{
    out = nullptr;
    outIsVolume3D = false;

    const QString norm = QDir::fromNativeSeparators(filePath);
    qInfo() << "[DCMTK] readSingleFileViaDCMTK =" << norm;

    // Enable DCMTK JPEG / JPEG-LS (idempotent).
    DJDecoderRegistration::registerCodecs();
    DJLSDecoderRegistration::registerCodecs();
    auto cleanupDecoders = []() {
        DJLSDecoderRegistration::cleanup();
        DJDecoderRegistration::cleanup();
    };

    // --- Load the DICOM file ---
    DcmFileFormat ff;
    const OFCondition st = ff.loadFile(norm.toUtf8().constData());
    if (st.bad()) {
        qWarning() << "[DCMTK] loadFile failed:" << st.text();
        cleanupDecoders();
        return false;
    }

    DcmDataset* ds = ff.getDataset();
    if (!ds) {
        qWarning() << "[DCMTK] No dataset";
        cleanupDecoders();
        return false;
    }

    // --- Transfer syntax info + request an uncompressed rep if needed ---
    const E_TransferSyntax origTS = ds->getOriginalXfer();
    qInfo() << "[DCMTK] Original TS =" << DcmXfer(origTS).getXferName()
            << " encapsulated=" << (DcmXfer(origTS).isEncapsulated() ? "yes" : "no");

    {
        const DcmRepresentationParameter* rp = nullptr;
        OFCondition cr = ds->chooseRepresentation(EXS_LittleEndianExplicit, rp);
        if (cr.bad()) cr = ds->chooseRepresentation(EXS_LittleEndianImplicit, rp);
        qDebug() << "[DCMTK] chooseRepresentation ->" << cr.text();
    }

    // --- Geometry tags ---
    const int rows   = getIntOr(ds, DCM_Rows,    -1);
    const int cols   = getIntOr(ds, DCM_Columns, -1);
    int       frames = getIntOr(ds, DCM_NumberOfFrames, 1);
    if (rows <= 0 || cols <= 0) {
        qWarning() << "[DCMTK] Invalid rows/cols:" << rows << cols;
        cleanupDecoders();
        return false;
    }
    if (frames <= 0) frames = 1;

    double sx = 1.0, sy = 1.0, sz = 1.0;
    double ox = 0.0, oy = 0.0, oz = 0.0;

    // Pixel Spacing (row, col) → (Y, X)
    {
        OFString ps;
        if (ds->findAndGetOFStringArray(DCM_PixelSpacing, ps).good()) {
            const auto parts = QString::fromLatin1(ps.c_str()).split('\\', Qt::SkipEmptyParts);
            if (parts.size() >= 2) {
                bool okr=false, okc=false;
                sy = parts[0].toDouble(&okr);
                sx = parts[1].toDouble(&okc);
                if (!okr) sy = 1.0;
                if (!okc) sx = 1.0;
            }
        }
        // Spacing between slices (fallback to thickness)
        sz = getDoubleOr(ds, DCM_SpacingBetweenSlices, 0.0);
        if (sz <= 0.0) sz = getDoubleOr(ds, DCM_SliceThickness, 1.0);
    }

    // Image Position (Patient) → origin
    {
        OFString ipp;
        if (ds->findAndGetOFStringArray(DCM_ImagePositionPatient, ipp).good()) {
            const auto v = QString::fromLatin1(ipp.c_str()).split('\\', Qt::SkipEmptyParts);
            if (v.size() >= 3) {
                bool okx=false, oky=false, okz=false;
                ox = v[0].toDouble(&okx);
                oy = v[1].toDouble(&oky);
                oz = v[2].toDouble(&okz);
                if (!okx) ox=0.0; if (!oky) oy=0.0; if (!okz) oz=0.0;
            }
        }
    }

    // --- Pixel type and fetch raw pixels from the (now) uncompressed rep ---
    const int bitsAllocated = getIntOr(ds, DCM_BitsAllocated, 16);
    const int pixelRep      = getIntOr(ds, DCM_PixelRepresentation, 0); // 0=unsigned, 1=signed

    const Uint8*  p8  = nullptr;
    const Uint16* p16 = nullptr;
    unsigned long count = static_cast<unsigned long>(rows) * cols * frames;
    OFCondition gotPixels = EC_IllegalCall;

    if (bitsAllocated <= 8) {
        const Uint8* tmp = nullptr;
        gotPixels = ds->findAndGetUint8Array(DCM_PixelData, tmp);
        if (gotPixels.good() && tmp) p8 = tmp; else count = 0;
    } else {
        const Uint16* tmp = nullptr;
        gotPixels = ds->findAndGetUint16Array(DCM_PixelData, tmp);
        if (gotPixels.good() && tmp) p16 = tmp; else count = 0;
    }

    if (!gotPixels.good() || (!p8 && !p16) || count == 0) {
        qWarning() << "[DCMTK] Could not access raw pixel array after chooseRepresentation()."
                   << "If this study is compressed with an unsupported codec (e.g. JPEG2000),"
                   << "decompress it (e.g. dcmdjpeg/dcmdjpls/dcmcjpeg) or add a JP2K decoder.";
        cleanupDecoders();
        return false;
    }

    // --- Build vtkImageData ---
    vtkSmartPointer<vtkImageData> img = vtkSmartPointer<vtkImageData>::New();
    img->SetExtent(0, cols - 1, 0, rows - 1, 0, frames - 1);
    img->SetSpacing(sx, sy, (frames > 1 ? sz : 1.0));
    img->SetOrigin(ox, oy, oz);

    if (bitsAllocated <= 8) {
        img->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
        std::memcpy(img->GetScalarPointer(), p8, static_cast<size_t>(count));
        qInfo() << "[DCMTK] Copied unsigned 8-bit pixels into vtkImageData";
    } else {
        const int vtkType = (pixelRep == 1 ? VTK_SHORT : VTK_UNSIGNED_SHORT);
        img->AllocateScalars(vtkType, 1);
        const size_t bytes = static_cast<size_t>(count) * sizeof(uint16_t);
        std::memcpy(img->GetScalarPointer(), p16, bytes);
        qInfo() << "[DCMTK] Copied" << (pixelRep ? "signed" : "unsigned")
                << "16-bit pixels into vtkImageData";
    }

    out = img;
    outIsVolume3D = (frames > 1);

    cleanupDecoders();
    return true;
}

bool DicomLoader::loadImage(const QString& inPath,
                            vtkSmartPointer<vtkImageData>& outImage,
                            bool& outIsVolume3D)
{
    qDebug() << "[Loader] loadImage(inPath) =" << inPath;
    outImage = nullptr;
    outIsVolume3D = false;

    QFileInfo fi(inPath);
    if (!fi.exists()) {
        qWarning() << "[Loader] Path does not exist:" << inPath;
        return false;
    }

    if (fi.isDir()) {
        if (tryVtkReadDir(inPath, outImage, outIsVolume3D)) return true;
        qWarning() << "[Loader] VTK dir read failed; DCMTK dir fallback not implemented yet.";
        return false;
    }

    // File: VTK first (fast), then robust DCMTK fallback
    if (tryVtkReadFile(inPath, outImage, outIsVolume3D)) return true;
    return readSingleFileViaDCMTK(inPath, outImage, outIsVolume3D);
}
