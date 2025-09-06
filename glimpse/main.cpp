// main.cpp
#include <QApplication>
#include "MainWindow.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    qInfo() << "[App] Qt + DCMTK bootstrap";
    MainWindow w; w.resize(1024,768); w.show();

    if (argc > 1) w.openDicomAt(QString::fromLocal8Bit(argv[1]));
    else          w.openDicomAt(R"(C:\datasets\MRI_DICOM\IMG-0001-00001_fromSimens.dcm)");

    return app.exec();
}
