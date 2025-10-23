#pragma once

#include <memory>
#include <vector>

#include <QString>
#include <QStringList>
#include <QVector>
#include <QByteArray>

#include <opencv2/core.hpp>
#include <QPointer>

// Forward declarations
class MainWindow;
struct DicomDll;           // FIX: match your dicom_dll.hpp (struct not class)
class ProgressSplash;

// Probe result type
#include "../model/io.hpp"   // io::ProbeResult

class AppController {
public:
    explicit AppController(MainWindow* view);
    ~AppController();

    // Entry points
    void load(const QString& pathQ);
    void show();                   // lightweight: delegates to doShowNow()
    void savePNG(const QString& outPath);
    void saveDICOM(const QString& outPath);

    // NEW: save a single .dcm with many frames (multiframe SC grayscale 8-bit)
    void saveDICOMMultiframe(const QString& outPath, int rows, int cols,
                             const QVector<QByteArray>& frames);

    // View callbacks
    void onSliceChanged(int idx);
    void onStartOverRequested();
    void applyNegative();
    void toggleNegative();

    // External C-ABI engine progress → UI
    void postSplashUpdateFromEngineThread(int pct, const QString& stage);

private:
    struct BusyScope {
        explicit BusyScope(MainWindow* v, const QString& message);
        ~BusyScope();
        MainWindow* v_ = nullptr;
    };

    // Pipeline pieces
    void clearLoadState();
    bool load_dicom(const std::string& path);
    bool reconstructAllSlicesFromDll(const QString& pathQ, bool fftshift);
    void prepare_fallback();

    // Show helpers
    void doShowNow();
    void show_metadata_and_image(const QStringList& meta, const cv::Mat& u8);
    void show_gradient_with_meta(const QStringList& meta);
    void showSlices(const std::vector<cv::Mat>& frames);
    void showSlice(int idx);
    void adoptReconStackF32(const std::vector<float>& stack, int S, int H, int W);

    // Local converters/utilities
    static cv::Mat to_u8(const cv::Mat& f32);
    static cv::Mat vecf32_to_u8(const std::vector<float>& v, int H, int W);
    static cv::Mat make_gradient(int H, int W);
    bool m_negativeMode = false;

    // Base (non-negative) pixels used to restore when toggling OFF
    std::vector<cv::Mat> m_slices8_base;
    cv::Mat              m_display8_base;

    // Helpers
    static cv::Mat invert8u(const cv::Mat& src);
    void captureNegativeBaseIfNeeded();
    void closeSplashIfAny();

private:
    // MVC
    MainWindow* m_view = nullptr;

    // Probe + metadata
    io::ProbeResult m_probe;
    QStringList     m_meta;

    // DICOM DLL wrapper (no global)
    std::unique_ptr<DicomDll> m_dicom;

    // Display state
    cv::Mat              m_display8;      // single image
    cv::Mat              m_lastImg8;      // last shown (for save)
    std::vector<cv::Mat> m_slices8;       // multi-slice stack
    int                  m_currentSlice = 0;

    // Source path (for UI messages)
    QString              m_sourcePathQ;

    // Splash (frameless progress window)
    QPointer<ProgressSplash> m_splash;
};
