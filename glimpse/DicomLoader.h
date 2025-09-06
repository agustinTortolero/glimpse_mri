#pragma once
#include <QImage>
#include <QString>

struct DicomLoader {
    // Existing: loads frame 0 (kept for backward compatibility)
    static QImage load(const QString& path);

    // NEW: how many frames/slices are in this DICOM?
    static int countFrames(const QString& path);

    // NEW: load a specific frameIndex in [0, count-1]
    static QImage loadFrame(const QString& path, int frameIndex);
    static QString info(const QString& path);
};
