#pragma once

#include <QString>
#include <QStringList>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

// Forward-declare to avoid heavy include; only pointers/refs here.
class MainWindow;

// These two are used BY VALUE as members, so we need their full types.
// If you prefer to keep the header light, you can change the members to pointers
// and forward-declare instead. For now we match your .cpp (by-value).
#include "../model/io.hpp"           // io::ProbeResult

class AppController {
public:
    ~AppController();

    explicit AppController(MainWindow* view);

    // Entry points
    void load(const QString& pathQ);
    void show();

    // Save actions (invoked from MainWindow context menu)
    void savePNG(const QString& outPath);
    void saveDICOM(const QString& outPath);

private:
    struct MriEngineDll;
    // -------- helper scope to show busy UI while doing work --------
    struct BusyScope {
        explicit BusyScope(MainWindow* v, const QString& message);
        ~BusyScope();
    private:
        MainWindow* v_ = nullptr;
    };

    // -------- optional dynamic loader for legacy GPU path --------
    struct MriEngineDll;                    // defined in .cpp (optional)
    std::unique_ptr<MriEngineDll> m_dll;    // not used if you go direct-API

    // ---- pipeline pieces (decl only; impl in .cpp) ----
    void clearLoadState();
    void load_probe(const std::string& path);
    bool load_dicom(const std::string& path);
    bool load_hdf5(const std::string& path);

    bool try_gpu_recon();                   // legacy dynamic-load path (optional)
    bool use_embedded_preview();
    void prepare_fallback();

    // Display helpers
    void doShowNow();
    void show_metadata_and_image(const QStringList& meta, const cv::Mat& u8);
    void show_gradient_with_meta(const QStringList& meta);

    // Multi-slice UI helpers
    void showSlices(const std::vector<cv::Mat>& slices);
    void showSlice(int idx);
    void onSliceChanged(int idx);           // connected from MainWindow

    // Data conversions
    static cv::Mat to_u8(const cv::Mat& f32);
    static cv::Mat vecf32_to_u8(const std::vector<float>& v, int H, int W);
    static cv::Mat make_gradient(int H, int W);

    // Adopt an S×H×W float stack into the GUI (builds 8-bit slices & slider)
    void adoptReconStackF32(const std::vector<float>& stack, int S, int H, int W);

private:
    // View
    MainWindow*   m_view = nullptr;

    // Metadata shown in the dock
    QStringList   m_meta;

    // What we learned during probing/loading
    io::ProbeResult m_probe;

    // K-space + optional embedded preview from loader
    std::vector<float> m_pre;
    int                m_preH = 0;
    int                m_preW = 0;

    // Last images to display/save
    cv::Mat              m_display8;     // single image path
    cv::Mat              m_lastImg8;     // last shown (for Save...)
    std::vector<cv::Mat> m_slices8;      // multi-slice path
    int                  m_currentSlice = 0;

    // Remember the source path (for UI)
    QString              m_sourcePathQ;
    bool reconstructAllSlicesFromDll(const QString& pathQ, bool fftshift);
};
