#pragma once
#include <QString>
#include <QStringList>
#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include "../model/mri_engine.hpp"
#include "../model/io.hpp"  // for io::ProbeResult, io::Flavor

#include "../view/mainwindow.hpp"
#include <QFileInfo>
#include <QTimer>
#include <chrono>

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


class MainWindow;

class AppController {
public:
    explicit AppController(MainWindow* view);

    // New API: decouple "load" from "show"
    void load(const QString& path);
    void show();

    // Backward compatibility
    void loadAndShow(const QString& path) {
        std::cerr << "[DBG][Controller] loadAndShow() [compat]\n";
        load(path);
        show();
    }

    void savePNG(const QString& outPath);
    void saveDICOM(const QString& outPath);

private:
    // ----------- Load pipeline -----------
    void clearLoadState();
    void load_probe(const std::string& path);
    bool load_dicom(const std::string& path);
    bool load_hdf5(const std::string& path);
    bool try_gpu_recon();           // uses m_ks → fills m_display8 if OK and tags meta
    bool use_embedded_preview();    // uses m_pre  → fills m_display8 if OK and tags meta
    void prepare_fallback();        // gradient display, with meta

    // ----------- Show pipeline -----------
    void show_metadata_and_image(const QStringList& meta, const cv::Mat& u8);
    void show_gradient_with_meta(const QStringList& meta);

    // ----------- Small helpers -----------
    static cv::Mat to_u8(const cv::Mat& f32);
    static cv::Mat vecf32_to_u8(const std::vector<float>& v, int H, int W);
    static cv::Mat make_gradient(int H, int W);
     void doShowNow(); // deferred UI push

private:
    MainWindow* m_view = nullptr;

    // Load state
    QString m_sourcePathQ;
    QStringList m_meta;          // accumulated metadata during load
    io::ProbeResult m_probe{};
    mri::KSpace m_ks;            // k-space buffer (if any)

    // Optional embedded preview from file loader
    std::vector<float> m_pre;
    int m_preH = 0, m_preW = 0;

    // Final 8-bit image prepared by the load stage and consumed by show()
    cv::Mat m_display8;

    // Also used by save operations
    cv::Mat m_lastImg8;

    // Unused from older design kept for compatibility with your project
    std::vector<float> m_image; // final float image (ny x nx)
};
