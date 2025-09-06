#include <QtCore>
#include <vector>
#include <algorithm>
#include <string>

// HDF5
#include <hdf5.h>

// DCMTK
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcuid.h>
#include <dcmtk/dcmdata/dcdeftag.h>

#ifndef UID_ROOT
// A private UID root for local testing. Replace with your org root if you have one.
#define UID_ROOT "1.2.826.0.1.3680043.8.498.999"
#endif

// ---------- Forward declarations ----------
static QStringList h5ListChildren(const QString& h5Path, const char* groupPath);
static bool h5ObjectExists(const QString& h5Path, const QString& objPath, bool mustBeDataset);
static bool readH5DatasetToFloat(const QString& h5Path, const QString& dsPath,
                                 std::vector<size_t>& dims, std::vector<float>& out);
static int handleFastMRI(const QString& h5Path, const QString& outBase);
static int handleISMRMRD(const QString& h5Path, const QString& outBase);
static int tryExportReconArrays(const QString& h5Path, const QString& outDir);

bool saveSliceAsDicomU16(const float* data, int rows, int cols,
                         float rescaleSlope, float rescaleIntercept,
                         int sliceIndex, const QString& outPath);

// ---------- Helpers ----------
static QString fmtDims(const std::vector<size_t>& d) {
    QStringList s; for (auto v : d) s << QString::number((qulonglong)v); return s.join("x");
}

static bool looksLikeImage(const std::vector<size_t>& dims, int& sliceAxis, int kMinImgDim = 32) {
    sliceAxis = -1;
    if (dims.size() == 2) {
        return int(dims[0]) >= kMinImgDim && int(dims[1]) >= kMinImgDim;
    }
    if (dims.size() == 3) {
        int s = 0; if (dims[1] < dims[s]) s = 1; if (dims[2] < dims[s]) s = 2;
        int a = (s + 1) % 3, b = (s + 2) % 3;
        if (int(dims[a]) >= kMinImgDim && int(dims[b]) >= kMinImgDim) { sliceAxis = s; return true; }
        return false;
    }
    return false;
}

// ---------- HDF5: list children of a group ----------
struct LinkIterCtx {
    std::vector<std::string> names;
};
static herr_t linkIterCb(hid_t group, const char* name, const H5L_info_t* /*info*/, void* op_data) {
    auto* ctx = reinterpret_cast<LinkIterCtx*>(op_data);
    if (name) ctx->names.emplace_back(name);
    return 0;
}
static QStringList h5ListChildren(const QString& h5Path, const char* groupPath) {
    QStringList out;
    hid_t file = H5Fopen(qPrintable(h5Path), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        qWarning() << "[WRN] H5Fopen failed for" << h5Path;
        return out;
    }
    hid_t grp = H5Gopen2(file, groupPath, H5P_DEFAULT);
    if (grp < 0) {
        H5Fclose(file);
        return out;
    }
    LinkIterCtx ctx;
    H5Literate(grp, H5_INDEX_NAME, H5_ITER_INC, nullptr, linkIterCb, &ctx);
    H5Gclose(grp);
    H5Fclose(file);
    for (const auto& s : ctx.names) out << QString::fromStdString(s);
    return out;
}

// ---------- HDF5: object exists ----------
static bool h5ObjectExists(const QString& h5Path, const QString& objPath, bool mustBeDataset) {
    bool ok = false;
    hid_t file = H5Fopen(qPrintable(h5Path), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) return false;
    if (!mustBeDataset) {
        htri_t e = H5Lexists(file, qPrintable(objPath), H5P_DEFAULT);
        ok = (e > 0);
    } else {
        hid_t ds = H5Dopen2(file, qPrintable(objPath), H5P_DEFAULT);
        if (ds >= 0) { ok = true; H5Dclose(ds); }
    }
    H5Fclose(file);
    return ok;
}

// ---------- HDF5: read any numeric dataset into vector<float> ----------
static bool readH5DatasetToFloat(const QString& h5Path, const QString& dsPath,
                                 std::vector<size_t>& dims, std::vector<float>& out) {
    dims.clear(); out.clear();
    hid_t file = H5Fopen(qPrintable(h5Path), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) { qWarning() << "[WRN] H5Fopen failed"; return false; }
    hid_t ds = H5Dopen2(file, qPrintable(dsPath), H5P_DEFAULT);
    if (ds < 0) { qWarning() << "[WRN] H5Dopen2 failed for" << dsPath; H5Fclose(file); return false; }

    hid_t space = H5Dget_space(ds);
    if (space < 0) { H5Dclose(ds); H5Fclose(file); return false; }
    int rank = H5Sget_simple_extent_ndims(space);
    if (rank <= 0) { H5Sclose(space); H5Dclose(ds); H5Fclose(file); return false; }
    std::vector<hsize_t> hdims(rank);
    H5Sget_simple_extent_dims(space, hdims.data(), nullptr);
    dims.resize(rank);
    for (int i = 0; i < rank; ++i) dims[i] = static_cast<size_t>(hdims[i]);

    size_t total = 1; for (auto d : dims) total *= d;
    out.resize(total);

    // Let HDF5 convert to float if needed
    herr_t st = H5Dread(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
    if (st < 0) {
        qWarning() << "[WRN] H5Dread failed for" << dsPath;
        H5Sclose(space); H5Dclose(ds); H5Fclose(file);
        out.clear(); dims.clear();
        return false;
    }

    H5Sclose(space);
    H5Dclose(ds);
    H5Fclose(file);
    return true;
}

// ---------- DICOM writer (U16) ----------
bool saveSliceAsDicomU16(const float* data, int rows, int cols,
                         float rescaleSlope, float rescaleIntercept,
                         int sliceIndex, const QString& outPath)
{
    // map to 12-bit in 16-bit container
    std::vector<Uint16> pix((size_t)rows * (size_t)cols);
    for (size_t i = 0; i < pix.size(); ++i) {
        double v = (double)data[i];
        double u = (v + (double)rescaleIntercept) * (double)rescaleSlope;
        if (u < 0.0) u = 0.0;
        if (u > 4095.0) u = 4095.0;
        pix[i] = (Uint16)(u + 0.5);
    }

    DcmFileFormat ff;
    DcmDataset* ds = ff.getDataset();

    // SOP
    ds->putAndInsertString(DCM_SOPClassUID, UID_SecondaryCaptureImageStorage);
    char sopUID[128]; dcmGenerateUniqueIdentifier(sopUID, UID_ROOT);
    ds->putAndInsertString(DCM_SOPInstanceUID, sopUID);

    // UIDs for Study/Series/FrameOfReference
    static QByteArray studyUID, seriesUID, forUID;
    if (studyUID.isEmpty()) { char u[128]; dcmGenerateUniqueIdentifier(u, UID_ROOT); studyUID = u; }
    if (seriesUID.isEmpty()) { char u[128]; dcmGenerateUniqueIdentifier(u, UID_ROOT); seriesUID = u; }
    if (forUID.isEmpty())   { char u[128]; dcmGenerateUniqueIdentifier(u, UID_ROOT); forUID   = u; }
    ds->putAndInsertString(DCM_StudyInstanceUID,  studyUID.constData());
    ds->putAndInsertString(DCM_SeriesInstanceUID, seriesUID.constData());
    ds->putAndInsertString(DCM_FrameOfReferenceUID, forUID.constData());

    // Date/Time
    const QString date = QDate::currentDate().toString("yyyyMMdd");
    const QString time = QTime::currentTime().toString("HHmmss");
    ds->putAndInsertString(DCM_StudyDate,  date.toUtf8().constData());
    ds->putAndInsertString(DCM_StudyTime,  time.toUtf8().constData());
    ds->putAndInsertString(DCM_SeriesDate, date.toUtf8().constData());
    ds->putAndInsertString(DCM_SeriesTime, time.toUtf8().constData());

    // Image Module (minimal)
    ds->putAndInsertString(DCM_Modality, "MR"); // or "OT" for SC; MR is fine for most viewers when minimal
    ds->putAndInsertString(DCM_SamplesPerPixel, "1");
    ds->putAndInsertString(DCM_PhotometricInterpretation, "MONOCHROME2");
    ds->putAndInsertUint16(DCM_Rows,    (Uint16)rows);
    ds->putAndInsertUint16(DCM_Columns, (Uint16)cols);
    ds->putAndInsertUint16(DCM_BitsAllocated, 16);
    ds->putAndInsertUint16(DCM_BitsStored,    12);
    ds->putAndInsertUint16(DCM_HighBit,       11);
    ds->putAndInsertUint16(DCM_PixelRepresentation, 0); // unsigned

    // Pixel spacing defaults; adjust if you know FOV/matrix
    ds->putAndInsertString(DCM_PixelSpacing, "1\\1");
    ds->putAndInsertString(DCM_SliceThickness, "1");

    // Instance number
    ds->putAndInsertUint16(DCM_InstanceNumber, (Uint16)(sliceIndex + 1));

    // Pixel data
    OFCondition cond = ds->putAndInsertUint16Array(DCM_PixelData, pix.data(), (Uint32)pix.size());
    if (cond.bad()) {
        qWarning() << "[WRN] putAndInsertUint16Array failed:" << cond.text();
        return false;
    }

    // Save
    OFCondition sc = ff.saveFile(outPath.toUtf8().constData(), EXS_LittleEndianExplicit);
    if (sc.bad()) {
        qWarning() << "[WRN] saveFile failed:" << sc.text();
        return false;
    }
    return true;
}

// ---------- fastMRI handler ----------
static int handleFastMRI(const QString& h5Path, const QString& outBase) {
    qDebug() << "[DBG] handleFastMRI ->" << h5Path;
    QString ds = h5ObjectExists(h5Path, "/reconstruction_rss", true) ? "/reconstruction_rss"
                 : h5ObjectExists(h5Path, "/reconstruction",    true) ? "/reconstruction"
                                                                     : QString();
    if (ds.isEmpty()) {
        qWarning() << "[WRN] fastMRI dataset not found (reconstruction_rss/reconstruction).";
        return 2;
    }

    std::vector<size_t> dims; std::vector<float> buf;
    if (!readH5DatasetToFloat(h5Path, ds, dims, buf)) return 3;

    qDebug() << "[DBG] Using dataset:" << ds;
    qDebug() << "[DBG] dataset rank=" << int(dims.size()) << " dims=" << fmtDims(dims);

    int S = 1, H = 0, W = 0;
    if (dims.size() == 3) { S = (int)dims[0]; H = (int)dims[1]; W = (int)dims[2]; }
    else if (dims.size() == 2) { S = 1; H = (int)dims[0]; W = (int)dims[1]; }
    else { qWarning() << "[WRN] Unexpected dims for fastMRI"; return 4; }

    if ((size_t)S * (size_t)H * (size_t)W != buf.size()) {
        qWarning() << "[WRN] Element count mismatch."; return 5;
    }

    // stats per slice (optional)
    for (int s = 0; s < S; ++s) {
        const float* p = buf.data() + (size_t)s * H * W;
        auto mm = std::minmax_element(p, p + (size_t)H * W);
        qDebug() << "[DBG] slice" << s << "min=" << *mm.first << "max=" << *mm.second;
    }

    QString outDir = outBase + "/fastMRI";
    QDir().mkpath(outDir);
    for (int s = 0; s < S; ++s) {
        const float* p = buf.data() + (size_t)s * H * W;
        auto mm = std::minmax_element(p, p + (size_t)H * W);
        float vmin = *mm.first, vmax = *mm.second;
        float slope = 4095.0f / (vmax - vmin + 1e-8f), inter = -vmin;
        QString fn = outDir + QString("/IMG_%1.dcm").arg(s, 4, 10, QLatin1Char('0'));
        qDebug() << "[DBG] save" << fn;
        if (!saveSliceAsDicomU16(p, H, W, slope, inter, s, fn))
            qWarning() << "[WRN] save failed for" << fn;
    }
    qDebug() << "[DBG] fastMRI DONE ->" << outDir;
    return 0;
}

// ---------- ISMRMRD handler ----------
static int handleISMRMRD(const QString& h5Path, const QString& outBase) {
    qDebug() << "[DBG] handleISMRMRD ->" << h5Path;

    // Header (optional)
    if (h5ObjectExists(h5Path, "/dataset/xml", true)) {
        std::vector<size_t> dims; std::vector<float> dummy;
        // Just show size: read raw string length via H5Dget_storage_size would be nice,
        // but keeping it simple: list children and log presence.
        qDebug() << "[DBG] XML header present (/dataset/xml)";
    }

    // Try image_* subgroups
    int totalWritten = 0;
    for (int i = 0; i < 32; ++i) {
        QString g = QString("/dataset/image_%1").arg(i);
        if (!h5ObjectExists(h5Path, g, /*mustBeDataset=*/false)) continue;
        // Common dataset name inside is usually "data"
        QString ds = g + "/data";
        if (!h5ObjectExists(h5Path, ds, true)) {
            qDebug() << "[DBG]" << g << "has no 'data' dataset.";
            continue;
        }
        std::vector<size_t> dims; std::vector<float> img;
        if (!readH5DatasetToFloat(h5Path, ds, dims, img)) continue;
        int sliceAxis = -1;
        if (!looksLikeImage(dims, sliceAxis)) continue;

        int H = 0, W = 0, S = 1;
        if (dims.size() == 2) { H = (int)dims[0]; W = (int)dims[1]; }
        else {
            int a = (sliceAxis + 1) % 3, b = (sliceAxis + 2) % 3;
            S = (int)dims[sliceAxis]; H = (int)dims[a]; W = (int)dims[b];
        }
        QString base = outBase + "/ISMRMRD/image_" + QString::number(i);
        QDir().mkpath(base);
        auto mm = std::minmax_element(img.begin(), img.end());
        float vmin = (mm.first != img.end()) ? *mm.first : 0.f;
        float vmax = (mm.second != img.end()) ? *mm.second : 1.f;
        float slope = 4095.0f / (vmax - vmin + 1e-8f), inter = -vmin;

        if (dims.size() == 2) {
            QString fn = base + "/IMG_0000.dcm";
            if (saveSliceAsDicomU16(img.data(), H, W, slope, inter, 0, fn)) totalWritten++;
        } else {
            // slice extract (generic)
            const size_t d0 = dims[0], d1 = dims[1], d2 = dims[2];
            auto at = [&](int s, int r, int c) -> float {
                if (sliceAxis == 0)      return img[(size_t)s * d1 * d2 + (size_t)r * d2 + (size_t)c];
                else if (sliceAxis == 1) return img[(size_t)r * d2 * d0 + (size_t)s * d2 + (size_t)c];
                else                     return img[(size_t)r * d2 + (size_t)c + (size_t)s * d0 * d1];
            };
            std::vector<float> slice((size_t)H * (size_t)W);
            for (int s = 0; s < S; ++s) {
                for (int r = 0; r < H; ++r)
                    for (int c = 0; c < W; ++c)
                        slice[(size_t)r * W + c] = at(s, r, c);
                QString fn = base + QString("/IMG_%1.dcm").arg(s, 4, 10, QLatin1Char('0'));
                if (saveSliceAsDicomU16(slice.data(), H, W, slope, inter, s, fn)) totalWritten++;
            }
        }
    }

    if (totalWritten == 0) {
        qWarning() << "[WRN] No ISMRMRD image_* groups found. Trying recon arrays (e.g., rec_mean, rec_std)...";
        int rc = tryExportReconArrays(h5Path, outBase);
        if (rc == 0) return 0;

        qWarning() << "[WRN] ISMRMRD handler found nothing exportable (rc=" << rc << ").";
        qWarning() << "[WRN] This file is probably acquisitions-only or stores images differently.";
        return 4;
    }

    qDebug() << "[DBG] ISMRMRD DONE ->" << (outBase + "/ISMRMRD") << "files=" << totalWritten;
    return 0;
}

// ---------- Recon-array fallback for ISMRMRD ----------
static int tryExportReconArrays(const QString& h5Path, const QString& outDir) {
    qDebug() << "[DBG] tryExportReconArrays scanning /dataset for numeric arrays...";
    int total = 0;

    QStringList children = h5ListChildren(h5Path, "/dataset");
    for (const QString& name : children) {
        if (name == "xml" || name == "data") continue; // skip header and raw k-space
        const QString dsPath = "/dataset/" + name;

        std::vector<size_t> dims;
        std::vector<float> buf;
        if (!readH5DatasetToFloat(h5Path, dsPath, dims, buf)) {
            qWarning() << "[WRN] readH5DatasetToFloat failed for" << dsPath;
            continue;
        }

        qDebug() << "[DBG]" << dsPath << "rank=" << int(dims.size()) << "dims=" << fmtDims(dims);

        int sliceAxis = -1;
        if (!looksLikeImage(dims, sliceAxis)) {
            qDebug() << "[SKIP]" << dsPath << "does not look like an image (too few/small dims)";
            continue;
        }

        int H = 0, W = 0, S = 1;
        if (dims.size() == 2) {
            H = (int)dims[0]; W = (int)dims[1]; sliceAxis = -1; S = 1;
        } else {
            int a = (sliceAxis + 1) % 3, b = (sliceAxis + 2) % 3;
            S = (int)dims[sliceAxis];
            H = (int)dims[a];
            W = (int)dims[b];
        }

        if ((size_t)H * (size_t)W * (size_t)std::max(1, S) != buf.size()) {
            qWarning() << "[WRN]" << dsPath << "element count mismatch, skip.";
            continue;
        }

        auto mm = std::minmax_element(buf.begin(), buf.end());
        float vmin = (mm.first != buf.end()) ? *mm.first : 0.f;
        float vmax = (mm.second != buf.end()) ? *mm.second : 1.f;
        float slope = 4095.0f / (vmax - vmin + 1e-8f), inter = -vmin;

        const QString subdir = outDir + "/ISMRMRD/" + name;
        QDir().mkpath(subdir);
        const QString base = subdir + "/IMG_";

        if (sliceAxis < 0) {
            QString fn = base + QString("%1.dcm").arg(0, 4, 10, QLatin1Char('0'));
            qDebug() << "[DBG] save" << fn << "min=" << vmin << "max=" << vmax
                     << "H=" << H << "W=" << W << "axis=" << sliceAxis;
            if (!saveSliceAsDicomU16(buf.data(), H, W, slope, inter, 0, fn))
                qWarning() << "[WRN] save failed for" << fn;
            else total++;
        } else {
            const size_t d0 = dims[0], d1 = dims[1], d2 = dims[2];
            auto at = [&](int s, int r, int c) -> float {
                if (sliceAxis == 0)      return buf[(size_t)s * d1 * d2 + (size_t)r * d2 + (size_t)c];
                else if (sliceAxis == 1) return buf[(size_t)r * d2 * d0 + (size_t)s * d2 + (size_t)c];
                else                     return buf[(size_t)r * d2 + (size_t)c + (size_t)s * d0 * d1];
            };

            std::vector<float> slice((size_t)H * (size_t)W);
            for (int s = 0; s < S; ++s) {
                for (int r = 0; r < H; ++r)
                    for (int c = 0; c < W; ++c)
                        slice[(size_t)r * W + c] = at(s, r, c);

                QString fn = base + QString("%1.dcm").arg(s, 4, 10, QLatin1Char('0'));
                qDebug() << "[DBG] save" << fn << "min=" << vmin << "max=" << vmax
                         << "H=" << H << "W=" << W << "axis=" << sliceAxis;
                if (!saveSliceAsDicomU16(slice.data(), H, W, slope, inter, s, fn))
                    qWarning() << "[WRN] save failed for" << fn;
                else total++;
            }
        }
    }

    // Helpful message if only /dataset/data (k-space) exists
    if (total == 0) {
        std::vector<size_t> dd; std::vector<float> tmp;
        if (readH5DatasetToFloat(h5Path, "/dataset/data", dd, tmp)) {
            qWarning() << "[WRN] /dataset/data present with dims =" << fmtDims(dd)
            << "-> this looks like raw k-space (acquisitions-only).";
            qWarning() << "[WRN] You need a reconstruction (e.g., Gadgetron or BART) to produce images.";
        } else {
            qWarning() << "[WRN] No exportable recon arrays found under /dataset.";
        }
    }

    if (total > 0) qDebug() << "[DBG] Recon-array export DONE ->" << (outDir + "/ISMRMRD") << "files=" << total;
    return (total > 0) ? 0 : 6;
}

// ---------- Format detection ----------
enum class H5Kind { FastMRI, ISMRMRD, Unknown };

// Helper to check a dataset exists
static inline bool hasDS(const QString& h5, const char* p) {
    return h5ObjectExists(h5, p, /*mustBeDataset=*/true);
}
// Helper to check a group exists
static inline bool hasGRP(const QString& h5, const char* p) {
    return h5ObjectExists(h5, p, /*mustBeDataset=*/false);
}

static H5Kind detectKind(const QString& h5Path) {
    // 1) fastMRI explicit signatures (check these *first*)
    if (hasDS(h5Path, "/reconstruction_rss") ||
        hasDS(h5Path, "/reconstruction")     ||
        hasDS(h5Path, "/kspace")             ||
        hasDS(h5Path, "/ismrmrd_header")) {
        qDebug() << "[DBG] detectKind: fastMRI signature found";
        return H5Kind::FastMRI;
    }

    // 2) ISMRMRD common layout
    if (hasGRP(h5Path, "/dataset")) {
        qDebug() << "[DBG] detectKind: /dataset present -> ISMRMRD";
        return H5Kind::ISMRMRD;
    }

    qDebug() << "[DBG] detectKind: Unknown layout";
    return H5Kind::Unknown;
}

// ---------- main ----------
int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    qDebug() << "[DBG] h5_dual_to_dicom starting. Args:" << QCoreApplication::arguments();

    // ---- Hardcoded test inputs (comment/uncomment if you like) ----
    const QString fastMRI_path =
        "C:/datasets/MRI_raw/FastMRI/brain_multicoil/file_brain_AXFLAIR_200_6002452.h5";
    const QString ismrmrd_path =
        "C:/datasets/MRI_raw/from mridata_dot_org/52c2fd53-d233-4444-8bfd-7c454240d314.h5";

    // Use CLI arg if given, else default to fastMRI
    QString input = (app.arguments().size() > 1) ? app.arguments().at(1) : ismrmrd_path;

    // Output base next to the chosen input
    QString outBase;
    {
        QFileInfo fi(input);
        QDir parent = fi.dir();
        outBase = parent.absoluteFilePath("dicom_out");
    }

    qDebug() << "[DBG] input =" << input;
    qDebug() << "[DBG] outBase=" << outBase;

    H5Kind kind = detectKind(input);
    if (kind == H5Kind::FastMRI) {
        qDebug() << "[DBG] Explicit fastMRI signature detected.";
        return handleFastMRI(input, outBase);
    } else if (kind == H5Kind::ISMRMRD) {
        qDebug() << "[DBG] Probing ISMRMRD:" << input;
        return handleISMRMRD(input, outBase);
    } else {
        qWarning() << "[WRN] Could not detect HDF5 kind (fastMRI/ISMRMRD).";
        return 10;
    }
}

