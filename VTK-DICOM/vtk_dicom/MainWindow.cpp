// MainWindow.cpp
#include "MainWindow.h"
#include "DicomLoader.h"

#include <QVTKOpenGLNativeWidget.h>
#include <QFileDialog>
#include <QMessageBox>
#include <QVBoxLayout>
#include <QMenuBar>
#include <QStatusBar>
#include <QDebug>
#include <QFileInfo>
#include <QDir>

// VTK
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkDICOMImageReader.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>
#include <vtkVolumeProperty.h>
#include <vtkVolume.h>
#include <vtkCamera.h>
#include <vtkImageShiftScale.h>
#include <vtkImageSliceMapper.h>
#include <vtkImageSlice.h>
#include <vtkStreamingDemandDrivenPipeline.h>
#include <vtkInformation.h> // <-- needed for GetOutputInformation(...)
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkRendererCollection.h>
#include <vtkFixedPointVolumeRayCastMapper.h> // <-- software volume mapper

#include <vtkImageProperty.h>


static QString prettyExtent(int ext[6]) {
    return QString("[%1,%2] x [%3,%4] x [%5,%6]")
    .arg(ext[0]).arg(ext[1]).arg(ext[2]).arg(ext[3]).arg(ext[4]).arg(ext[5]);
}

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    qInfo() << "[MainWindow] ctor";

    auto* central = new QWidget(this);
    auto* layout  = new QVBoxLayout(central);
    layout->setContentsMargins(0,0,0,0);

    m_vtk = new QVTKOpenGLNativeWidget(central);
    layout->addWidget(m_vtk);
    setCentralWidget(central);

    setupMenus();
    statusBar()->showMessage("Ready.");

    buildEmptyScene();
}

MainWindow::~MainWindow() { qInfo() << "[MainWindow] dtor"; }

void MainWindow::setupMenus() {
    auto* fileMenu = menuBar()->addMenu("&File");
    auto* openAct  = fileMenu->addAction("Open DICOM &Path...");
    openAct->setShortcut(QKeySequence::Open);
    connect(openAct, &QAction::triggered, this, &MainWindow::onOpen);

    fileMenu->addSeparator();
    auto* quitAct = fileMenu->addAction("E&xit");
    connect(quitAct, &QAction::triggered, this, &QWidget::close);
}

void MainWindow::buildEmptyScene() {
    qInfo() << "[VTK] building empty scene";
    auto renderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    m_vtk->setRenderWindow(renderWindow);

    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->SetBackground(0.05, 0.05, 0.08);
    renderWindow->AddRenderer(renderer);

    renderWindow->Render();
}

void MainWindow::onOpen() {
    QString path = QFileDialog::getExistingDirectory(this, "Select DICOM folder");
    if (path.isEmpty()) {
        // Allow single-file as fallback (will use its parent dir)
        path = QFileDialog::getOpenFileName(this, "Select a DICOM file", {}, "DICOM (*.dcm);;All (*.*)");
    }
    if (path.isEmpty()) return;
    openDicomAt(path);
}

void MainWindow::openDicomAt(const QString& path) {
    qInfo() << "[MainWindow] openDicomAt path=" << path;
    loadVolumeFromPath(path); // wrapper
}

// 1) WRAPPER that renders (kept)
void MainWindow::loadVolumeFromPath(const QString& path)
{
    qDebug() << "[Wrapper] loadVolumeFromPath(path) -> calling DicomLoader::loadImage";

    vtkSmartPointer<vtkImageData> image;
    bool isVolume3D = false;

    if (!DicomLoader::loadImage(path, image, isVolume3D) || !image)
    {
        qWarning() << "[Wrapper] loader failed or no image";
        QMessageBox::warning(this, "DICOM", "Failed to load DICOM image.");
        return;
    }

    // Get (or create) renderer
    auto* rw = m_vtk->renderWindow();
    vtkRenderer* ren = nullptr;
    if (rw->GetRenderers() && rw->GetRenderers()->GetNumberOfItems() > 0) {
        ren = rw->GetRenderers()->GetFirstRenderer();
    } else {
        auto r = vtkSmartPointer<vtkRenderer>::New();
        r->SetBackground(0.05, 0.05, 0.08);
        rw->AddRenderer(r);
        ren = r;
    }
    ren->RemoveAllViewProps();

    if (isVolume3D) {
        qDebug() << "[Wrapper] Building 3D volume pipeline (software mapper)";
        auto sh = vtkSmartPointer<vtkImageShiftScale>::New();
        sh->SetInputData(image);
        sh->SetShift(0.0);
        sh->SetScale(1.0);
        sh->SetOutputScalarTypeToUnsignedShort();
        sh->Update();

        auto mapper = vtkSmartPointer<vtkFixedPointVolumeRayCastMapper>::New();
        mapper->SetInputConnection(sh->GetOutputPort());

        // Range-aware TFs
        double r[2]; sh->GetOutput()->GetScalarRange(r);
        const double lo = r[0], hi = r[1];
        auto ctf = vtkSmartPointer<vtkColorTransferFunction>::New();
        ctf->AddRGBPoint(lo, 0.0, 0.0, 0.0);
        ctf->AddRGBPoint(hi, 1.0, 1.0, 1.0);

        auto otf = vtkSmartPointer<vtkPiecewiseFunction>::New();
        otf->AddPoint(lo, 0.00);
        otf->AddPoint((0.25*hi + 0.75*lo), 0.05);
        otf->AddPoint((0.50*hi + 0.50*lo), 0.15);
        otf->AddPoint(hi, 0.35);

        auto prop = vtkSmartPointer<vtkVolumeProperty>::New();
        prop->SetColor(ctf);
        prop->SetScalarOpacity(otf);
        prop->SetInterpolationTypeToLinear();
        prop->ShadeOn();

        auto vol = vtkSmartPointer<vtkVolume>::New();
        vol->SetMapper(mapper);
        vol->SetProperty(prop);
        ren->AddVolume(vol);
    } else {
        qDebug() << "[Wrapper] Building 2D slice pipeline";

        auto im = vtkSmartPointer<vtkImageSliceMapper>::New();
        im->SetInputData(image);
        im->SliceAtFocalPointOn();
        im->SliceFacesCameraOn();

        auto slice = vtkSmartPointer<vtkImageSlice>::New();
        slice->SetMapper(im);

        // AUTO window/level so it’s not white anymore
        double r[2]; image->GetScalarRange(r);
        const double window = std::max(1.0, r[1] - r[0]);
        const double level  = 0.5*(r[1] + r[0]);
        slice->GetProperty()->SetColorWindow(window);
        slice->GetProperty()->SetColorLevel(level);
        qDebug() << "[2D] auto WL  window=" << window << " level=" << level;

        ren->AddViewProp(slice);
    }

    ren->ResetCamera();
    rw->Render();
}

bool MainWindow::loadVolumeFromPath(const QString& inPath,
                                    vtkSmartPointer<vtkImageData>& outImage,
                                    bool& outIsVolume3D)
{
    qDebug() << "[Helper] loadVolumeFromPath(inPath, outImage, outIsVolume3D) called with" << inPath;

    // Always reset outputs first
    outImage = nullptr;
    outIsVolume3D = false;

    // Delegate the actual loading/decoding to our DicomLoader (VTK → DCMTK fallback).
    const bool ok = DicomLoader::loadImage(inPath, outImage, outIsVolume3D);
    if (!ok || !outImage)
    {
        qWarning() << "[Helper] DicomLoader::loadImage FAILED for path:" << inPath;
        return false;
    }

    // Debug: print some basics about the image we got back.
    int ext[6] = {0,0,0,0,0,0};
    outImage->GetExtent(ext);
    double sp[3] = {0,0,0};
    outImage->GetSpacing(sp);
    double range[2] = {0,0};
    outImage->GetScalarRange(range);

    qDebug() << "[Helper] extent =" << prettyExtent(ext);
    qDebug() << "[Helper] spacing=" << sp[0] << sp[1] << sp[2];
    qDebug() << "[Helper] scalar range =" << range[0] << "…" << range[1];
    qDebug() << "[Helper] outIsVolume3D =" << outIsVolume3D;

    return true;
}
