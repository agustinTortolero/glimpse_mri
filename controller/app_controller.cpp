#include "app_controller.hpp"
#include "../model/io_fastmri.hpp"
#include "../view/mainwindow.hpp"
#include "../src/image_utils.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>

AppController::AppController(MainWindow* view) : m_view(view) {}

void AppController::loadAndShow(const QString& h5_path_q)
{
    const std::string path = h5_path_q.toStdString();
    std::cerr << "[DBG][Controller] Loading fastMRI file: " << path << "\n";

    constexpr bool kPreferPreRecon = false; // <<<<<< set FALSE to force GPU recon

    std::vector<float> pre;
    int preH = 0, preW = 0;
    std::string dbg;

    if (!mri::load_fastmri_kspace(path, m_ks, &pre, &preH, &preW, &dbg)) {
        std::cerr << "[ERR][Controller] Failed to read kspace or preRecon.\n";
        m_view->setImage(imgutil::make_test_gradient(512, 512));
        return;
    }
    std::cerr << dbg;

    if (kPreferPreRecon && !pre.empty()) {
        std::cerr << "[DBG][Controller] Using reconstruction_rss from file (" << preW << "x" << preH << ").\n";
        m_view->setImage(imgutil::to_8u(pre, preH, preW));
        return;
    }

    std::cerr << "[DBG][Controller] kspace loaded. C=" << m_ks.coils
              << " ny=" << m_ks.ny << " nx=" << m_ks.nx << "\n";

    // GPU IFFT + RSS (will center-crop to square)
    std::string recon_dbg;
    if (!mri::ifft_rss_gpu(m_ks, m_image, &recon_dbg)) {
        std::cerr << "[ERR][Controller] Reconstruction failed.\n";
        m_view->setImage(imgutil::make_test_gradient(std::max(64, m_ks.ny), std::max(64, m_ks.nx)));
        return;
    }
    std::cerr << recon_dbg;

    // m_image already cropped to square (e.g., 320x320). Use mri_engine’s outH/outW or infer.
    const int outH = std::min(m_ks.ny, m_ks.nx);
    const int outW = outH;
    m_view->setImage(imgutil::to_8u(m_image, outH, outW));
    std::cerr << "[DBG][Controller] Image sent to view.\n";
}
