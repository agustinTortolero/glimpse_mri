#pragma once
#include <QMainWindow>
#include <QImage>
#include <QString>

class QLabel;
class QSlider;
class QScrollArea;
class QTextEdit;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    void openDicomAt(const QString& path);

protected:
    void contextMenuEvent(QContextMenuEvent* ev) override;
    void keyPressEvent(QKeyEvent* ev) override;
    void wheelEvent(QWheelEvent* ev) override;

private slots:
    void openDicom();
    void savePng();
    void saveBmp();
    void onSliceChanged(int);

private:
    void setImage(const QImage& img);
    void zoomBy(double factor, const QPointF& mousePosInViewport);
    void applyScale();
    void updateInfoPanel(const QString& path);   // NEW

    // Widgets
    QLabel*      imageLabel_  = nullptr;
    QScrollArea* scrollArea_  = nullptr;
    QSlider*     sliceSlider_ = nullptr;
    QTextEdit*   infoEdit_    = nullptr;         // NEW

    // Image & file state
    QImage  current_;
    QString currentPath_;
    int     frames_ = 1;
    int     frame_  = 0;

    // Zoom
    double  scale_    = 1.0;
    double  minScale_ = 0.1;
    double  maxScale_ = 8.0;
};
