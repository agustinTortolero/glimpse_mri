// main.cpp
#include <QApplication>
#include <QSurfaceFormat>
#include <QDebug>
#include <QFileInfo>

#include <QVTKOpenGLNativeWidget.h> // make sure Qt glue headers are visible
#include "MainWindow.h"

int main(int argc, char** argv) {
    // Ensure VTK-compatible default OpenGL format for the Qt widget
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());

    QApplication app(argc, argv);
    qInfo() << "[App] Qt + DCMTK + VTK bootstrap";

    MainWindow w; w.resize(1200, 900); w.show();

    QString argPath;
    if (argc > 1) {
        argPath = QString::fromLocal8Bit(argv[1]);
        qInfo() << "[App] Arg path =" << argPath;
    } else {
        // You can point this to a file OR a folder:
        // If a file is given, we use its parent dir as a DICOM series.
        argPath = R"(C:\datasets\MRI_DICOM\IMG-0001-00001_fromSimens.dcm)";
        qInfo() << "[App] Using default path =" << argPath;
    }

    w.openDicomAt(argPath);
    return app.exec();
}
