#pragma once
#include <QMainWindow>
#include <QDockWidget>
#include <QPlainTextEdit>
#include <QLabel>
#include <QSlider>
#include <QStringList>

#include <opencv2/core.hpp>

class QContextMenuEvent;
class QResizeEvent;

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);

    // ---- metadata panel ----
    void setMetadata(const QStringList& lines);
    void appendMetadataLine(const QString& line);

    // ---- image cycle ----
    void beginNewImageCycle();            // clears image & slider state
    void setImage(const cv::Mat& img8u);  // expects CV_8UC1
    void refreshPixmap();

    // ---- multi-slice helpers ----
    // nSlices >= 2 => enable + show slider with range [0..nSlices-1], else disable/hide
    void enableSliceSlider(int nSlices);
    // programmatic sync (blocks signals to avoid feedback loops)
    void setSliceIndex(int idx);

    // ---- busy UI ----
    void beginBusy(const QString& message);
    void endBusy();

signals:
    void requestSavePNG(const QString& outPath);
    void requestSaveDICOM(const QString& outPath);
    void sliceChanged(int idx);

protected:
    void contextMenuEvent(QContextMenuEvent* ev) override;
    void resizeEvent(QResizeEvent* ev) override;
    bool eventFilter(QObject* obj, QEvent* ev) override;

private slots:
    void onSavePNG();
    void onSaveDICOM();

private:
    // widgets
    QLabel*         m_label       = nullptr;
    QPlainTextEdit* m_metaText    = nullptr;
    QDockWidget*    m_metaDock    = nullptr;
    QSlider*        m_sliceSlider = nullptr;

    // state
    bool    m_hasImage    = false;
    cv::Mat m_img8;                 // currently shown 8-bit image
    int     m_busyNesting = 0;

    void drawSliceOverlay(cv::Mat& img8);
};
