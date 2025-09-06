#include "utils.h"
#include <algorithm>
#include <cstdint>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcuid.h>
#include <dcmtk/ofstd/ofuuid.h>

void scaleToU16(const float* src, uint16_t* dst, int n,
                float* outMin, float* outMax) {
    float mn = std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        float v = src[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    if (outMin) *outMin = mn;
    if (outMax) *outMax = mx;

    if (!(mx > mn)) {
        qWarning() << "[WRN] scaleToU16: degenerate range; writing zeros";
        std::fill(dst, dst + n, 0);
        return;
    }
    const double scale = 65535.0 / double(mx - mn);
    for (int i = 0; i < n; ++i) {
        double dv = (double(src[i]) - double(mn)) * scale;
        if (dv < 0) dv = 0;
        if (dv > 65535.0) dv = 65535.0;
        dst[i] = static_cast<uint16_t>(dv + 0.5);
    }
}


bool saveSliceAsDicomU16(const QString& outPath,
                         int width,
                         int height,
                         const std::vector<uint16_t>& pixels)
{
    qDebug() << "[DBG] saveSliceAsDicomU16 ->" << outPath
             << "W=" << width << "H=" << height
             << "px=" << pixels.size();

    if (width <= 0 || height <= 0) {
        qWarning() << "[WRN] Invalid image size.";
        return false;
    }
    const size_t expected = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (pixels.size() < expected) {
        qWarning() << "[WRN] Pixel buffer too small:" << pixels.size() << "expected" << expected;
        return false;
    }

    DcmFileFormat file;
    DcmDataset* ds = file.getDataset();

    // Basic patient/study/series
    ds->putAndInsertString(DCM_PatientName, "Anon^Patient");
    ds->putAndInsertString(DCM_PatientID,   "0001");
    ds->putAndInsertString(DCM_Modality,    "MR");

    // UIDs
    char studyUID[128], seriesUID[128], sopUID[128];
    dcmGenerateUniqueIdentifier(studyUID);
    dcmGenerateUniqueIdentifier(seriesUID);
    dcmGenerateUniqueIdentifier(sopUID);

    ds->putAndInsertString(DCM_StudyInstanceUID,  studyUID);
    ds->putAndInsertString(DCM_SeriesInstanceUID, seriesUID);
    ds->putAndInsertString(DCM_SOPClassUID,       UID_SecondaryCaptureImageStorage);
    ds->putAndInsertString(DCM_SOPInstanceUID,    sopUID);

    // Image pixel module
    ds->putAndInsertUint16(DCM_SamplesPerPixel,       1);
    ds->putAndInsertString(DCM_PhotometricInterpretation, "MONOCHROME2");
    ds->putAndInsertUint16(DCM_Rows,                  static_cast<Uint16>(height));
    ds->putAndInsertUint16(DCM_Columns,               static_cast<Uint16>(width));
    ds->putAndInsertUint16(DCM_BitsAllocated,         16);
    ds->putAndInsertUint16(DCM_BitsStored,            12);
    ds->putAndInsertUint16(DCM_HighBit,               11);
    ds->putAndInsertUint16(DCM_PixelRepresentation,   0); // 0 = unsigned

    // Optional defaults (can be overridden by your higher-level caller)
    ds->putAndInsertString(DCM_PatientOrientation, "");
    ds->putAndInsertString(DCM_ImageType, "DERIVED\\SECONDARY");
    ds->putAndInsertString(DCM_LossyImageCompression, "00");
    // If you know spacing/thickness/orientation, set (0028,0030), (0018,0050), (0020,0037) etc.

    // Pixel data
    const Uint16* ptr = reinterpret_cast<const Uint16*>(pixels.data());
    const unsigned long count = static_cast<unsigned long>(expected);
    OFCondition cond = ds->putAndInsertUint16Array(DCM_PixelData, ptr, count);
    if (cond.bad()) {
        qWarning() << "[ERR] putAndInsertUint16Array failed:" << cond.text();
        return false;
    }

    // Save Explicit VR Little Endian
    OFCondition sc = file.saveFile(outPath.toUtf8().constData(), EXS_LittleEndianExplicit);
    if (sc.bad()) {
        qWarning() << "[ERR] saveFile failed:" << sc.text();
        return false;
    }

    qDebug() << "[DBG] DICOM saved.";
    return true;
}

bool saveSliceAsDicomU16(const QString& outPath,
                         const float* img, int w, int h, int sliceNo,
                         const QString& studyLabel,
                         const QString& seriesLabel) {
    qDebug() << "[DBG] DICOM save ->" << outPath
             << " w=" << w << " h=" << h << " slice=" << sliceNo;

    std::vector<uint16_t> u16(size_t(w) * size_t(h));
    float mn=0.0f, mx=0.0f;
    scaleToU16(img, u16.data(), int(u16.size()), &mn, &mx);

    char studyUID[128] = {0};
    char seriesUID[128] = {0};
    char sopInstUID[128] = {0};

    dcmGenerateUniqueIdentifier(studyUID,  nullptr);
    dcmGenerateUniqueIdentifier(seriesUID, nullptr);
    dcmGenerateUniqueIdentifier(sopInstUID,nullptr);

    DcmFileFormat ff;
    DcmDataset* ds = ff.getDataset();

    // Minimal required tags
    ds->putAndInsertOFStringArray(DCM_SpecificCharacterSet, "ISO_IR 100");
    ds->putAndInsertOFStringArray(DCM_SOPClassUID, UID_SecondaryCaptureImageStorage);
    ds->putAndInsertOFStringArray(DCM_SOPInstanceUID, sopInstUID);

    ds->putAndInsertOFStringArray(DCM_PatientName, "Anonymous");
    ds->putAndInsertOFStringArray(DCM_PatientID,   "000000");

    ds->putAndInsertOFStringArray(DCM_StudyInstanceUID,  studyUID);
    ds->putAndInsertOFStringArray(DCM_SeriesInstanceUID, seriesUID);

    ds->putAndInsertOFStringArray(DCM_StudyDescription,  studyLabel.toUtf8().constData());
    ds->putAndInsertOFStringArray(DCM_SeriesDescription, seriesLabel.toUtf8().constData());

    // Image Pixel module
    ds->putAndInsertUint16(DCM_SamplesPerPixel, 1);
    ds->putAndInsertOFStringArray(DCM_PhotometricInterpretation, "MONOCHROME2");
    ds->putAndInsertUint16(DCM_Rows,    static_cast<Uint16>(h));
    ds->putAndInsertUint16(DCM_Columns, static_cast<Uint16>(w));
    ds->putAndInsertUint16(DCM_BitsAllocated,    16);
    ds->putAndInsertUint16(DCM_BitsStored,       16);
    ds->putAndInsertUint16(DCM_HighBit,          15);
    ds->putAndInsertUint16(DCM_PixelRepresentation, 0); // unsigned
    ds->putAndInsertUint16(DCM_SmallestImagePixelValue, 0);
    ds->putAndInsertUint16(DCM_LargestImagePixelValue, 65535);

    // Optional windowing from mn/mx
    ds->putAndInsertFloat64(DCM_WindowCenter,  (mx + mn) * 0.5);
    ds->putAndInsertFloat64(DCM_WindowWidth,   std::max(1.0, double(mx - mn)));

    // Instance number
    ds->putAndInsertSint32(DCM_InstanceNumber, sliceNo + 1);

    // Pixel data
    OFCondition st = ds->putAndInsertUint16Array(DCM_PixelData, u16.data(), u16.size());
    if (st.bad()) {
        qWarning() << "[ERR] putAndInsertUint16Array failed:" << st.text();
        return false;
    }

    // Write (explicit little endian)
    OFCondition sv = ff.saveFile(outPath.toUtf8().constData(), EXS_LittleEndianExplicit);
    if (sv.bad()) {
        qWarning() << "[ERR] DICOM save failed:" << sv.text();
        return false;
    }
    return true;
}
