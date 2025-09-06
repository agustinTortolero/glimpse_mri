#pragma once
#include <QtCore>
#include <vector>
#include <hdf5.h>

enum class H5Kind { FastMRI, ISMRMRD, Unknown };

bool h5ObjectExists(const QString& filePath, const char* path, bool mustBeDataset);
QStringList h5ListChildren(const QString& filePath, const char* groupPath);

bool readH5DatasetToFloat(const QString& filePath, const QString& dsPath,
                          std::vector<size_t>& dims, std::vector<float>& out);

bool h5DatasetIsNumeric(hid_t file, const QString& path);

H5Kind detectKind(const QString& h5Path);

QString shapeToString(const std::vector<size_t>& dims);
