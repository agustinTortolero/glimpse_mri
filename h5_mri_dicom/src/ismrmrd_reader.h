#pragma once
#include <QtCore>
#include <QString>

// Export ISMRMRD images or recon arrays (if image_* not present)
int exportIsmrmrdToDicom(const QString& h5Path, const QString& outBase);


// 3-arg convenience wrapper (what your call site uses)
bool exportNumericArrayAsImages(const QString& filePath,
                                const QString& datasetPath,
                                const QString& outDir);

// Detailed overload (lets you choose slice axis and scaling behavior)
bool exportNumericArrayAsImages(const QString& filePath,
                                const QString& datasetPath,
                                const QString& outDir,
                                int sliceAxis,          // -1 = auto (prefer [S,H,W] => axis 0)
                                bool clampToU16);       // true = scale per-slice to 0..4095
