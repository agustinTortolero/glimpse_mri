#pragma once
#include <QString>
#include <vtkSmartPointer.h>
class vtkImageData;

class DicomLoader {
public:
    // If a FILE is passed, we use its parent dir for series-based readers.
    static QString normalizeDicomPath(const QString& inPath);

    // High-level: try VTK; if it fails with compressed data, fall back to DCMTK.
    static bool loadImage(const QString& inPath,
                          vtkSmartPointer<vtkImageData>& outImage,
                          bool& outIsVolume3D);

private:
    // Helpers
    static bool tryVtkReadFile(const QString& filePath,
                               vtkSmartPointer<vtkImageData>& out,
                               bool& outIsVolume3D);

    static bool tryVtkReadDir(const QString& dirPath,
                              vtkSmartPointer<vtkImageData>& out,
                              bool& outIsVolume3D);

    // DCMTK single-file (handles multi-frame too)
    static bool readSingleFileViaDCMTK(const QString& filePath,
                                       vtkSmartPointer<vtkImageData>& out,
                                       bool& outIsVolume3D);
};
