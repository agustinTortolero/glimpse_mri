#pragma once
#include <QtCore>
#include <vector>

struct Slice2D {
    int w = 0, h = 0, index = 0;
    std::vector<float> pixels; // row-major
};

// DICOM export (DCMTK)
bool saveSliceAsDicomU16(const QString& outPath,
                         const float* img, int w, int h, int sliceNo,
                         const QString& studyLabel,
                         const QString& seriesLabel);

// Basic scaling
void scaleToU16(const float* src, uint16_t* dst, int n,
                float* outMin = nullptr, float* outMax = nullptr);


bool saveSliceAsDicomU16(const QString& outPath,
                         int width,
                         int height,
                         const std::vector<uint16_t>& pixels);
