#pragma once
#include <QMainWindow>
#include <QPointer>

#include <vtkSmartPointer.h>
class vtkImageData;

class QVTKOpenGLNativeWidget;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent=nullptr);
    ~MainWindow() override;

    // Open file or directory. If a file is passed, we’ll try its parent dir as a series.
    void openDicomAt(const QString& path);

private slots:
    void onOpen();

private:
    void setupMenus();
    void buildEmptyScene();

    // Returns true if load succeeded, populates outImage and outIsVolume3D.
    bool loadVolumeFromPath(const QString& path,
                            vtkSmartPointer<vtkImageData>& outImage,
                            bool& outIsVolume3D);
    // MainWindow.h (add this next to the existing declaration)
    void loadVolumeFromPath(const QString& path);  // <-- wrapper overload

    // already present:
    // bool loadVolumeFromPath(const QString& path,
    //                         vtkSmartPointer<vtkImageData>& outImage,
    //                         bool& outIsVolume3D);


private:
    QPointer<QVTKOpenGLNativeWidget> m_vtk;
};
