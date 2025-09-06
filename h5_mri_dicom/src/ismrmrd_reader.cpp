#include "ismrmrd_reader.h"
#include "read_hdf5.h"
#include "utils.h"
#include <algorithm>

#include <limits>

static bool looksImageLike(const std::vector<size_t>& dims, int minDim=16) {
    int big = 0;
    for (size_t d : dims) if (int(d) >= minDim) ++big;
    return (dims.size() >= 2 && big >= 2);
}

int exportNumericArrayAsImages(const QString& outDir,
                               const QString& name,
                               const std::vector<size_t>& dims,
                               const std::vector<float>& data,
                               int preferredSliceAxis /* -1 auto, else 0/1/2 */)
{
    qDebug() << "[DBG] exportNumericArrayAsImages name=" << name
             << "dims="
             << ([&]{
                    QStringList ss;
                    for (size_t d : dims) ss << QString::number(qulonglong(d));
                    return ss.join('x');
                })();

    if (data.empty() || dims.empty()) {
        qWarning() << "[WRN] exportNumericArrayAsImages: empty data or dims.";
        return 0;
    }

    // Heuristics for image-like arrays
    auto looks_like_image_dims = [](size_t h, size_t w) -> bool {
        return (h >= 16 && w >= 16); // super simple guard
    };

    QString destDir = outDir + "/" + name;
    QDir().mkpath(destDir);

    int written = 0;

    if (dims.size() == 2) {
        // 2D -> single slice
        const size_t H = dims[0], W = dims[1];
        if (!looks_like_image_dims(H, W)) {
            qWarning() << "[SKIP] 2D array does not look like an image (H=" << H << ", W=" << W << ")";
            return 0;
        }
        if (data.size() != H * W) {
            qWarning() << "[WRN] Size mismatch: data=" << data.size() << " H*W=" << (H*W);
            return 0;
        }

        // Min/max for slice
        float lo = std::numeric_limits<float>::infinity();
        float hi = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < data.size(); ++i) {
            float v = data[i];
            if (std::isnan(v) || std::isinf(v)) continue;
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        if (!std::isfinite(lo) || !std::isfinite(hi) || hi <= lo) {
            // fallback
            lo = 0.0f; hi = 1.0f;
        }

        qDebug() << "[DBG] 2D min=" << lo << "max=" << hi << " H=" << H << " W=" << W;

        std::vector<uint16_t> out(static_cast<size_t>(H * W));
        const double denom = std::max(1e-12, double(hi - lo));
        for (size_t i = 0; i < data.size(); ++i) {
            double v = std::isfinite(data[i]) ? (data[i] - lo) / denom : 0.0;
            if (v < 0.0) v = 0.0;
            if (v > 1.0) v = 1.0;
            out[i] = static_cast<uint16_t>(std::llround(v * 4095.0)); // 12-bit range
        }

        QString f = QString("%1/IMG_%2.dcm").arg(destDir).arg(0, 4, 10, QChar('0'));
        qDebug() << "[DBG] save" << f;
        if (saveSliceAsDicomU16(f, int(W), int(H), out)) ++written;
        return written;
    }

    if (dims.size() == 3) {
        size_t d0 = dims[0], d1 = dims[1], d2 = dims[2];

        // Choose slice axis
        int axis = preferredSliceAxis;
        if (axis < 0 || axis > 2) {
            if (looks_like_image_dims(d1, d2)) axis = 0;            // (S,H,W)
            else if (looks_like_image_dims(d0, d2)) axis = 1;       // (H,S,W)
            else if (looks_like_image_dims(d0, d1)) axis = 2;       // (H,W,S)
            else {
                qWarning() << "[SKIP] 3D dims not image-like:" << d0 << d1 << d2;
                return 0;
            }
        }

        size_t S = 0, H = 0, W = 0;
        if (axis == 0) { S = d0; H = d1; W = d2; }
        else if (axis == 1) { S = d1; H = d0; W = d2; }
        else /*axis==2*/ { S = d2; H = d0; W = d1; }

        if (!looks_like_image_dims(H, W)) {
            qWarning() << "[SKIP] chosen HxW not image-like: H=" << H << " W=" << W;
            return 0;
        }

        // Precompute strides for row-major layout
        size_t stride0 = d1 * d2;
        size_t stride1 = d2;
        size_t stride2 = 1;

        auto index = [&](size_t i0, size_t i1, size_t i2) -> size_t {
            // linear index for (i0,i1,i2) with dims (d0,d1,d2)
            return i0 * stride0 + i1 * stride1 + i2 * stride2;
        };

        qDebug() << "[DBG] 3D axis=" << axis << " slices=" << S << " H=" << H << " W=" << W;

        std::vector<uint16_t> out(static_cast<size_t>(H * W));

        for (size_t s = 0; s < S; ++s) {
            // Compute min/max for this slice
            double lo = std::numeric_limits<double>::infinity();
            double hi = -std::numeric_limits<double>::infinity();

            for (size_t y = 0; y < H; ++y) {
                for (size_t x = 0; x < W; ++x) {
                    size_t idx;
                    if (axis == 0)      idx = index(s, y, x);        // (S,H,W)
                    else if (axis == 1) idx = index(y, s, x);        // (H,S,W)
                    else                idx = index(y, x, s);        // (H,W,S)

                    float v = data[idx];
                    if (!std::isfinite(v)) continue;
                    if (v < lo) lo = v;
                    if (v > hi) hi = v;
                }
            }
            if (!std::isfinite(lo) || !std::isfinite(hi) || hi <= lo) {
                lo = 0.0; hi = 1.0;
            }

            // Scale slice to 12-bit
            const double denom = std::max(1e-12, hi - lo);
            for (size_t y = 0; y < H; ++y) {
                for (size_t x = 0; x < W; ++x) {
                    size_t idx;
                    if (axis == 0)      idx = index(s, y, x);
                    else if (axis == 1) idx = index(y, s, x);
                    else                idx = index(y, x, s);

                    double v = std::isfinite(data[idx]) ? (data[idx] - lo) / denom : 0.0;
                    if (v < 0.0) v = 0.0;
                    if (v > 1.0) v = 1.0;
                    out[y * W + x] = static_cast<uint16_t>(std::llround(v * 4095.0));
                }
            }

            // Save
            QString f = QString("%1/IMG_%2.dcm").arg(destDir).arg(qulonglong(s), 4, 10, QChar('0'));
            qDebug() << "[DBG] slice" << s << "min=" << lo << "max=" << hi << "save" << f;
            if (saveSliceAsDicomU16(f, int(W), int(H), out))
                ++written;
        }

        qDebug() << "[DBG] exportNumericArrayAsImages wrote" << written << "slices to" << destDir;
        return written;
    }

    // N-D fallback: treat last two dims as H,W and flatten the rest into slices
    if (dims.size() >= 4) {
        size_t nd = dims.size();
        size_t H = dims[nd - 2];
        size_t W = dims[nd - 1];
        if (!looks_like_image_dims(H, W)) {
            qWarning() << "[SKIP] N-D dims; last two not image-like: H=" << H << " W=" << W;
            return 0;
        }

        size_t tile = H * W;
        size_t S = 1;
        for (size_t i = 0; i < nd - 2; ++i) S *= dims[i];
        if (data.size() != S * tile) {
            qWarning() << "[WRN] N-D mismatch: data=" << data.size()
            << " expected=" << (S*tile);
            return 0;
        }

        qDebug() << "[DBG] N-D collapse: S=" << S << " H=" << H << " W=" << W;

        std::vector<uint16_t> out(tile);
        for (size_t s = 0; s < S; ++s) {
            const float* sliceBase = data.data() + s * tile;

            double lo = std::numeric_limits<double>::infinity();
            double hi = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < tile; ++i) {
                float v = sliceBase[i];
                if (!std::isfinite(v)) continue;
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            }
            if (!std::isfinite(lo) || !std::isfinite(hi) || hi <= lo) { lo = 0.0; hi = 1.0; }

            const double denom = std::max(1e-12, hi - lo);
            for (size_t i = 0; i < tile; ++i) {
                double v = std::isfinite(sliceBase[i]) ? (sliceBase[i] - lo) / denom : 0.0;
                if (v < 0.0) v = 0.0;
                if (v > 1.0) v = 1.0;
                out[i] = static_cast<uint16_t>(std::llround(v * 4095.0));
            }

            QString f = QString("%1/IMG_%2.dcm").arg(destDir).arg(qulonglong(s), 4, 10, QChar('0'));
            qDebug() << "[DBG] slice" << s << "min=" << lo << "max=" << hi << "save" << f;
            if (saveSliceAsDicomU16(f, int(W), int(H), out))
                ++written;
        }
        return written;
    }

    qWarning() << "[SKIP] Unsupported dims rank =" << dims.size();
    return 0;
}

int exportIsmrmrdToDicom(const QString& h5Path, const QString& outBase) {
    qDebug() << "[DBG] handleISMRMRD ->" << h5Path;

    if (!h5ObjectExists(h5Path, "/dataset", false)) {
        qWarning() << "[WRN] /dataset group not found; not an ISMRMRD file.";
        return 4;
    }

    // 1) Try image_* groups (by name existence). Placeholders for future.
    for (int i = 0; i < 32; ++i) {
        const QString grp = QString("/dataset/image_%1").arg(i);
        if (h5ObjectExists(h5Path, grp.toUtf8().constData(), false)) {
            qDebug() << "[DBG] Found group" << grp << "(not exported in this minimal demo)";
        }
    }

    // 2) Scan for simple numeric arrays like /dataset/recon, /dataset/rec_mean, ...
    qDebug() << "[DBG] tryExportReconArrays scanning /dataset for numeric arrays...";
    QStringList kids = h5ListChildren(h5Path, "/dataset");

    // open once to check dtypes
    hid_t file = H5Fopen(h5Path.toUtf8().constData(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) { qWarning() << "[WRN] H5Fopen failed"; return 6; }

    const QString outDirBase = outBase + "/ISMRMRD";
    QDir().mkpath(outDirBase);
    int imagesExported = 0;

    for (const QString& k : kids) {
        const QString path = "/dataset/" + k;
        if (!h5DatasetIsNumeric(file, path)) {
            qDebug() << "[SKIP]" << path << "is not a numeric dataset (compound/VLEN/other)";
            continue;
        }
        imagesExported += exportNumericArrayAsImages(h5Path, path, outDirBase);
    }

    H5Fclose(file);

    if (imagesExported == 0) {
        qWarning() << "[WRN] No exportable recon arrays found under /dataset.";
        return 6;
    }

    qDebug() << "[DBG] Recon-array export DONE ->" << outDirBase << "files=" << imagesExported;
    return 0;
}

// --- 3-arg wrapper: keeps existing call sites working ---
bool exportNumericArrayAsImages(const QString& filePath,
                                const QString& datasetPath,
                                const QString& outDir)
{
    qDebug() << "[DBG] exportNumericArrayAsImages(wrapper)" << filePath << datasetPath << "->" << outDir;
    // Defaults that work for most ISMRMRD recon arrays: [S,H,W] with S as slice axis
    return exportNumericArrayAsImages(filePath, datasetPath, outDir,
                                      /*sliceAxis*/ -1, /*clampToU16*/ true);
}

// --- Full implementation ---
static inline QString leafName(const QString& h5Path) {
    // last component after '/'
    int p = h5Path.lastIndexOf('/');
    return (p >= 0) ? h5Path.mid(p + 1) : h5Path;
}

static void quantizeToU16_12bit(const float* src, size_t n,
                                std::vector<uint16_t>& dst,
                                float& outMin, float& outMax)
{
    // Per-slice min/max
    outMin = std::numeric_limits<float>::infinity();
    outMax = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < n; ++i) {
        if (src[i] < outMin) outMin = src[i];
        if (src[i] > outMax) outMax = src[i];
    }
    if (!(outMax > outMin)) { // handle NaNs / flat images
        outMin = 0.0f; outMax = 1.0f;
    }
    const float scale = (outMax > outMin) ? (4095.0f / (outMax - outMin)) : 0.0f;

    dst.resize(n);
    for (size_t i = 0; i < n; ++i) {
        float v = (src[i] - outMin) * scale;
        // clamp 0..4095 (12-bit)
        if (v < 0.0f) v = 0.0f;
        if (v > 4095.0f) v = 4095.0f;
        dst[i] = static_cast<uint16_t>(std::lround(v));
    }
}

bool exportNumericArrayAsImages(const QString& filePath,
                                const QString& datasetPath,
                                const QString& outDir,
                                int sliceAxis,
                                bool clampToU16)
{
    qDebug() << "[DBG] exportNumericArrayAsImages(full)" << filePath << datasetPath
             << "out=" << outDir << "axis=" << sliceAxis << "clamp=" << clampToU16;

    std::vector<size_t> dims;
    std::vector<float> data;
    if (!readH5DatasetToFloat(filePath, datasetPath, dims, data)) {
        qWarning() << "[WRN] readH5DatasetToFloat failed for" << datasetPath;
        return false;
    }

    if (dims.size() < 2 || dims.size() > 4) {
        qWarning() << "[WRN] Unsupported rank for export:" << int(dims.size())
        << "dims=" << QString::fromStdString([&]{
            QString s; for (size_t i=0;i<dims.size();++i){ if(i) s+="_"; s+=QString::number(qulonglong(dims[i])); }
            return s;
        }().toStdString());
        return false;
    }

    // Heuristic: last two dims are H,W; earlier dims collapse into "slices"
    size_t H = dims[dims.size()-2];
    size_t W = dims[dims.size()-1];

    size_t slices = 1;
    for (size_t i = 0; i + 2 < dims.size(); ++i) slices *= dims[i];

    if (H < 8 || W < 8) {
        qWarning() << "[WRN] Not image-like (H or W too small): H=" << H << "W=" << W;
        return false;
    }

    const size_t pixelsPerSlice = H * W;
    const size_t totalNeeded = pixelsPerSlice * slices;
    if (data.size() < totalNeeded) {
        qWarning() << "[WRN] Data too small: have" << data.size() << "need" << totalNeeded;
        return false;
    }

    // Prepare output directory
    QDir().mkpath(outDir);
    const QString baseName = leafName(datasetPath);
    const QString outSubDir = QDir(outDir).filePath(baseName);
    QDir().mkpath(outSubDir);

    qDebug() << "[DBG]" << datasetPath << "rank=" << int(dims.size())
             << "dims=" << QString("%1x%2").arg(qlonglong(H)).arg(qlonglong(W))
             << "slices=" << qlonglong(slices);

    // Iterate slices in row-major order of the leading dims
    std::vector<uint16_t> u16;
    size_t written = 0;
    for (size_t s = 0; s < slices; ++s) {
        const float* src = data.data() + s * pixelsPerSlice;

        float mn=0.f, mx=0.f;
        quantizeToU16_12bit(src, pixelsPerSlice, u16, mn, mx);
        qDebug() << "[DBG] slice" << qlonglong(s)
                 << "min=" << mn << "max=" << mx;

        const QString outPath = QDir(outSubDir).filePath(
            QString("IMG_%1.dcm").arg(int(s), 4, 10, QChar('0')));

        if (!saveSliceAsDicomU16(outPath, int(W), int(H), u16)) {
            qWarning() << "[WRN] saveSliceAsDicomU16 failed for" << outPath;
            continue;
        }
        ++written;
    }

    qDebug() << "[DBG] exportNumericArrayAsImages DONE ->" << outSubDir
             << "files=" << qlonglong(written);
    return (written > 0);
}
