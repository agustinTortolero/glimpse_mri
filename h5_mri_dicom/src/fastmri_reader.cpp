#include "fastmri_reader.h"
#include "read_hdf5.h"
#include <algorithm>

static bool pickFastMriDataset(const QString& h5, QString& ds) {
    // preference order
    const char* candidates[] = {
        "/reconstruction_rss",
        "/reconstruction",
        "/recons",
        "/reconstruction_esc",
        "/images"
    };
    for (auto* c : candidates) {
        if (h5ObjectExists(h5, c, true)) { ds = c; return true; }
    }
    // last resort: kspace is NOT image; we skip it here
    return false;
}

int exportFastMRIToDicom(const QString& h5Path, const QString& outBase) {
    qDebug() << "[DBG] handleFastMRI ->" << h5Path;

    QString dsPath;
    if (!pickFastMriDataset(h5Path, dsPath)) {
        qWarning() << "[WRN] No image-like dataset in fastMRI file.";
        return 2;
    }
    qDebug() << "[DBG] Using dataset:" << dsPath;

    std::vector<size_t> dims;
    std::vector<float> data;
    if (!readH5DatasetToFloat(h5Path, dsPath, dims, data)) {
        qWarning() << "[WRN] Failed reading" << dsPath;
        return 3;
    }
    qDebug() << "[DBG] dataset rank=" << int(dims.size()) << " dims= \"" << shapeToString(dims) << "\"";

    if (dims.size() == 2) {
        // single 2D image
        const int H = int(dims[0]), W = int(dims[1]);
        const QString outDir = outBase + "/fastMRI";
        QDir().mkpath(outDir);
        const QString out = outDir + "/IMG_0000.dcm";
        qDebug() << "[DBG] save" << out;
        saveSliceAsDicomU16(out, data.data(), W, H, 0, "fastMRI", "reconstruction");
        return 0;
    }

    if (dims.size() == 3) {
        const int S = int(dims[0]);
        const int H = int(dims[1]);
        const int W = int(dims[2]);
        const size_t sliceElems = size_t(H) * size_t(W);
        const QString outDir = outBase + "/fastMRI";
        QDir().mkpath(outDir);

        for (int s = 0; s < S; ++s) {
            const float* img = data.data() + size_t(s) * sliceElems;
            const QString out = outDir + QString("/IMG_%1.dcm").arg(s, 4, 10, QChar('0'));

            auto mm = std::minmax_element(img, img + sliceElems);
            qDebug() << "[DBG] slice" << s
                     << "min=" << *mm.first << "max=" << *mm.second;
            qDebug() << "[DBG] save" << out;
            saveSliceAsDicomU16(out, img, W, H, s, "fastMRI", "reconstruction");
        }
        qDebug() << "[DBG] fastMRI DONE ->" << outDir;
        return 0;
    }

    qWarning() << "[WRN] fastMRI dataset has unexpected rank; skipping.";
    return 4;
}
