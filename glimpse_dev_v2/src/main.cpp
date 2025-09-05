#include <QApplication>
#include "../view/mainwindow.hpp"
#include "../controller/app_controller.hpp"
#include <iostream>

// --- Hardcoded paths per iteration spec ---
// (DICOM) — active for this iteration
//static const char* MRIPath = "C:\\\\datasets\\\\MRI_DICOM\\\\IMG-0001-00001_fromSimens.dcm";

// (fastMRI) — keep for later testing
 //static const char* MRIPath = "C:\\\\datasets\\\\MRI_raw\\\\FastMRI\\\\brain_multicoil\\\\file_brain_AXFLAIR_200_6002452.h5";

// (ISMRMRD) — keep for later testing
static const char* MRIPath = "C:\\\\datasets\\\\MRI_raw\\\\from mridata_dot_org\\\\52c2fd53-d233-4444-8bfd-7c454240d314.h5";

int main(int argc, char** argv)
{
    try {
        std::cerr << "[DBG] Qt app starting.\n";
        QApplication app(argc, argv);
        MainWindow w;
        AppController controller(&w);

        std::cerr << "[DBG] Loading hardcoded DICOM file:\n  " << MRIPath << "\n";
        controller.loadAndShow(QString::fromUtf8(MRIPath));

        w.setWindowTitle("Glimpse MRI — Read DICOM (Iteration 2)");
        w.show();
        return app.exec();
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n";
        return 1;
    }
}
