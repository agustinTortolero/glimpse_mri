


/*
  h5_mri_dicom — Qt/MSVC demo helpers for reading MRI HDF5 and exporting/viewing.

  What it does:
    - Detects HDF5 “flavors”: fastMRI (reconstruction_rss) and ISMRMRD (/dataset).
    - Loads image-like arrays (float32), rescales per-slice to uint16, and
      writes classic single-frame DICOMs via DCMTK (one file per slice).
    - Emits verbose [DBG]/[WRN] logs for every step to aid debugging.

  Libraries:
    - HDF5 (via vcpkg), DCMTK, ISMRMRD.

  Notes:
    - ISMRMRD acquisitions-only files are not reconstructed here; a separate
      recon step (e.g., Gadgetron/BART/FFTW) is needed before export.
    - Geometry tags (orientation/spacing) are placeholders; fill from metadata
      or recon outputs when available.

  Modules:
    - h5_reader.*      : small HDF5 utilities (open, dims, dtype, read arrays).
    - fastmri_reader.* : fastMRI reader/exporter (uses /reconstruction_rss).
    - ismrmrd_reader.* : ISMRMRD reader; tries image_* or recon arrays.
    - utils.*          : DICOM write (DCMTK), scaling, paths, UID/time helpers.
    - main.cpp         : example wiring with hardcoded test paths.

  Intended use:
    - Integrate into a Qt app that reads HDF5, (optionally) reconstructs,
      and displays MRI images; this demo provides I/O + DICOM export scaffolding.
*/
#include <QtCore>
#include <hdf5.h>
#include "read_hdf5.h"
#include "fastmri_reader.h"
#include "ismrmrd_reader.h"

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    qDebug() << "[DBG] h5_dual_to_dicom starting. Args:" << app.arguments();

    // Silence HDF5 error stack during probes
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

    // ---- hardcoded inputs (comment one if you like) ----
    const QString fastMRIPath = "C:/datasets/MRI_raw/FastMRI/brain_multicoil/file_brain_AXFLAIR_200_6002452.h5";
    const QString mridataPath = "C:/datasets/MRI_raw/from mridata_dot_org/52c2fd53-d233-4444-8bfd-7c454240d314.h5";

    // Choose one here:
    const QString input = mridataPath;
    // const QString input = mridataPath;

    const QString outBase = QFileInfo(input).absolutePath() + "/dicom_out";
    qDebug() << "[DBG] input =" << input;
    qDebug() << "[DBG] outBase=" << outBase;

    // Detect kind (fastMRI first)
    H5Kind kind = detectKind(input);
    int rc = 0;

    switch (kind) {
        case H5Kind::FastMRI:
            qDebug() << "[DBG] Explicit fastMRI signature detected.";
            rc = exportFastMRIToDicom(input, outBase);
            break;
        case H5Kind::ISMRMRD:
            qDebug() << "[DBG] ISMRMRD layout detected.";
            rc = exportIsmrmrdToDicom(input, outBase);
            break;
        default:
            qWarning() << "[WRN] Unknown HDF5 layout; nothing to do.";
            rc = 1;
            break;
    }

    return rc;
}
