#pragma once
#include <QtCore>
#include "utils.h"

// Export fastMRI to DICOM (reads /reconstruction_rss preferred)
int exportFastMRIToDicom(const QString& h5Path, const QString& outBase);
