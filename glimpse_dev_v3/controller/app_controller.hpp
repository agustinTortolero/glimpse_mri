// app_controller.hpp
#pragma once
#include <QString>
#include <QStringList>
#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include "../model/mri_engine.hpp"
#include "../model/io.hpp"
#include "../view/mainwindow.hpp"
#include <QFileInfo>
#include <QTimer>
#include <chrono>
#include <memory>          // <— add
#include <QLibrary>        // <— add

class MainWindow;  // forward-declare only

namespace {
struct BusyScope {
    MainWindow* view;
    explicit BusyScope(MainWindow* v, const QString& msg) : view(v) {
        if (view) view->beginBusy(msg);
    }
    ~BusyScope() {
        if (view) view->endBusy();
    }
};
} // namespace

// --- NEW: a minimal dynamic loader for the MRI engine DLL ---
struct MriEngineDll {
    // C-ABI signatures we expect from the DLL (exported with extern "C")
    using PFN_init = void (*)(int device); // mri_engine_init(int)
    using PFN_ifft = bool (*)(             // mri_ifft_rss_interleaved(...)
        const float* kspace_cplx2,         // interleaved (real,imag)
        int coils, int ny, int nx,
        float* out_rss,                    // output buffer (H*W floats)
        int* outH, int* outW,
        char* dbg_buf, int dbg_cap         // optional debug text
        );

    QLibrary lib;
    PFN_init p_init = nullptr;
    PFN_ifft p_ifft = nullptr;
    bool loaded = false;

    bool load(const QString& explicitPath = QString());
};

class AppController {
public:
    explicit AppController(MainWindow* view);

    void load(const QString& path);
    void show();
    void loadAndShow(const QString& path) {
        std::cerr << "[DBG][Controller] loadAndShow() [compat]\n";
        load(path);
        show();
    }

    void savePNG(const QString& outPath);
    void saveDICOM(const QString& outPath);

private:
    void clearLoadState();
    void load_probe(const std::string& path);
    bool load_dicom(const std::string& path);
    bool load_hdf5(const std::string& path);
    bool try_gpu_recon();
    bool use_embedded_preview();
    void prepare_fallback();

    void show_metadata_and_image(const QStringList& meta, const cv::Mat& u8);
    void show_gradient_with_meta(const QStringList& meta);
    void doShowNow();

    // helpers
    static cv::Mat to_u8(const cv::Mat& f32);
    static cv::Mat vecf32_to_u8(const std::vector<float>& v, int H, int W);
    static cv::Mat make_gradient(int H, int W);

private:
    MainWindow* m_view = nullptr;

    QString m_sourcePathQ;
    QStringList m_meta;
    io::ProbeResult m_probe{};
    mri::KSpace m_ks;

    std::vector<float> m_pre;
    int m_preH = 0, m_preW = 0;

    cv::Mat m_display8;
    cv::Mat m_lastImg8;

    std::vector<float> m_image;

    // --- NEW: dynamically loaded engine
    std::unique_ptr<MriEngineDll> m_dll;   // lazily created on first use
#ifdef MRI_INPROC_FALLBACK
    bool recon_inproc();  // NEW: compiled only if you define MRI_INPROC_FALLBACK
#endif
};
