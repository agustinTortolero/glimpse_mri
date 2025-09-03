#include <QApplication>
#include "../view/mainwindow.hpp"
#include "../controller/app_controller.hpp"
#include <iostream>



// Hardcoded path per first-iteration spec
static const char* kFastMRIPath =
  "C:\\\\datasets\\\\MRI_raw\\\\FastMRI\\\\brain_multicoil\\\\file_brain_AXFLAIR_200_6002452.h5";

int main(int argc, char** argv)
{
  try {
    std::cerr << "[DBG] Qt app starting.\n";
    QApplication app(argc, argv);
    MainWindow w;
    AppController controller(&w);

    std::cerr << "[DBG] Loading hardcoded file:\n  " << kFastMRIPath << "\n";
    controller.loadAndShow(QString::fromUtf8(kFastMRIPath));

    w.setWindowTitle("Glimpse MRI — FFT/RSS (Iteration 1)");
    w.show();
    return app.exec();
  } catch (const std::exception& e) {
    std::cerr << "[FATAL] " << e.what() << "\n";
    return 1;
  }
}
