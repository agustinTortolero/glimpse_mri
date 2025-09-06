#include "read_hdf5.h"
#include <algorithm>

QString shapeToString(const std::vector<size_t>& dims) {
    QString s;
    for (size_t i = 0; i < dims.size(); ++i) {
        s += QString::number(qulonglong(dims[i]));
        if (i + 1 < dims.size()) s += 'x';
    }
    return s;
}

bool h5ObjectExists(const QString& filePath, const char* path, bool mustBeDataset) {
    hid_t f = H5Fopen(filePath.toUtf8().constData(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) return false;
    bool ok = false;
    if (mustBeDataset) {
        hid_t d = H5Dopen2(f, path, H5P_DEFAULT);
        ok = (d >= 0);
        if (d >= 0) H5Dclose(d);
    } else {
        hid_t g = H5Gopen2(f, path, H5P_DEFAULT);
        if (g >= 0) { ok = true; H5Gclose(g); }
        if (!ok) {
            hid_t d = H5Dopen2(f, path, H5P_DEFAULT);
            if (d >= 0) { ok = true; H5Dclose(d); }
        }
    }
    H5Fclose(f);
    return ok;
}

QStringList h5ListChildren(const QString& filePath, const char* groupPath) {
    QStringList out;
    hid_t f = H5Fopen(filePath.toUtf8().constData(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) {
        qWarning() << "[WRN] H5Fopen failed for" << filePath;
        return out;
    }
    hid_t g = H5Gopen2(f, groupPath, H5P_DEFAULT);
    if (g < 0) {
        H5Fclose(f);
        qWarning() << "[WRN] H5Gopen2 failed for" << groupPath;
        return out;
    }
    H5G_info_t gi;
    H5Gget_info(g, &gi);
    for (hsize_t i = 0; i < gi.nlinks; ++i) {
        ssize_t len = H5Lget_name_by_idx(g, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
        if (len < 0) continue;
        QByteArray buf;
        buf.resize(int(len) + 1);
        H5Lget_name_by_idx(g, ".", H5_INDEX_NAME, H5_ITER_INC, i, buf.data(), size_t(buf.size()), H5P_DEFAULT);
        buf[len] = '\0';
        out << QString::fromUtf8(buf.constData());
    }
    H5Gclose(g);
    H5Fclose(f);
    return out;
}

static bool isNumericClass(H5T_class_t cls) {
    return (cls == H5T_FLOAT || cls == H5T_INTEGER);
}

bool h5DatasetIsNumeric(hid_t file, const QString& path) {
    hid_t d = H5Dopen2(file, path.toUtf8().constData(), H5P_DEFAULT);
    if (d < 0) return false;
    hid_t t = H5Dget_type(d);
    H5T_class_t cls = H5Tget_class(t);
    bool ok = isNumericClass(cls);
    H5Tclose(t);
    H5Dclose(d);
    return ok;
}

bool readH5DatasetToFloat(const QString& filePath, const QString& dsPath,
                          std::vector<size_t>& dims, std::vector<float>& out) {
    hid_t f = H5Fopen(filePath.toUtf8().constData(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) { qWarning() << "[WRN] H5Fopen failed"; return false; }
    hid_t d = H5Dopen2(f, dsPath.toUtf8().constData(), H5P_DEFAULT);
    if (d < 0) { H5Fclose(f); qWarning() << "[WRN] H5Dopen2 failed for" << dsPath; return false; }

    hid_t t = H5Dget_type(d);
    H5T_class_t cls = H5Tget_class(t);
    if (!isNumericClass(cls)) {
        qWarning() << "[WRN] Dataset is not numeric:" << dsPath;
        H5Tclose(t); H5Dclose(d); H5Fclose(f);
        return false;
    }

    hid_t s = H5Dget_space(d);
    int rank = H5Sget_simple_extent_ndims(s);
    if (rank <= 0) { H5Sclose(s); H5Tclose(t); H5Dclose(d); H5Fclose(f); return false; }
    std::vector<hsize_t> hdims(rank);
    H5Sget_simple_extent_dims(s, hdims.data(), nullptr);
    dims.resize(rank);
    size_t count = 1;
    for (int i = 0; i < rank; ++i) {
        dims[i] = size_t(hdims[i]);
        count *= size_t(hdims[i]);
    }
    out.resize(count);

    herr_t st = H5Dread(d, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
    if (st < 0) {
        qWarning() << "[WRN] H5Dread failed for" << dsPath;
        H5Sclose(s); H5Tclose(t); H5Dclose(d); H5Fclose(f);
        return false;
    }

    H5Sclose(s);
    H5Tclose(t);
    H5Dclose(d);
    H5Fclose(f);
    return true;
}

H5Kind detectKind(const QString& h5Path) {
    // fastMRI signatures FIRST
    if (h5ObjectExists(h5Path, "/reconstruction_rss", true) ||
        h5ObjectExists(h5Path, "/reconstruction",     true) ||
        h5ObjectExists(h5Path, "/kspace",             true) ||
        h5ObjectExists(h5Path, "/ismrmrd_header",     true)) {
        qDebug() << "[DBG] detectKind: fastMRI signature found";
        return H5Kind::FastMRI;
    }
    // ISMRMRD layout
    if (h5ObjectExists(h5Path, "/dataset", false)) {
        qDebug() << "[DBG] detectKind: /dataset present -> ISMRMRD";
        return H5Kind::ISMRMRD;
    }
    qDebug() << "[DBG] detectKind: Unknown layout";
    return H5Kind::Unknown;
}
