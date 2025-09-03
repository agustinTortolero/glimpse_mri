#pragma once
#include <QString>
#include <vector>
#include "../model/mri_engine.hpp"

class MainWindow;

class AppController {
public:
    explicit AppController(MainWindow* view);
    void loadAndShow(const QString& h5_path);

private:
    MainWindow* m_view = nullptr;
    mri::KSpace m_ks;
    std::vector<float> m_image; // final float image (ny x nx)
};
