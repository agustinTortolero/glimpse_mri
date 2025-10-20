#pragma once
#include <QMainWindow>
#include <QStringList>
#include <opencv2/core.hpp>

// Forward decls (keep header light)
class QLabel;
class QDockWidget;
class QPlainTextEdit;
class QSlider;
class QEvent;
class QDragEnterEvent;
class QDragMoveEvent;
class QDropEvent;
class QContextMenuEvent;   // for contextMenuEvent override
class QResizeEvent;        // for resizeEvent override
class QUrl;

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent=nullptr);

    void setMetadata(const QStringList& lines);
    void appendMetadataLine(const QString& line);
    void beginNewImageCycle();
    void setImage(const cv::Mat& img8u);
    void enableSliceSlider(int nSlices);
    void setSliceIndex(int idx);


signals:
    void requestSavePNG(const QString& outPath);
    void requestSaveDICOM(const QString& outPath);
    void sliceChanged(int idx);
    void fileDropped(const QString& path);   // DnD
    void startOverRequested();

protected:
    bool eventFilter(QObject* obj, QEvent* ev) override;

    // Drag & drop handlers
    void dragEnterEvent(QDragEnterEvent* ev) override;
    void dragMoveEvent(QDragMoveEvent* ev) override;
    void dropEvent(QDropEvent* ev) override;

    // Declare these overrides (implemented in .cpp)
    void contextMenuEvent(QContextMenuEvent* ev) override;
    void resizeEvent(QResizeEvent* ev) override;

public slots:
    // Busy UI used by controller’s current BusyScope
    void beginBusy(const QString& message);
    void endBusy();

private:
    void refreshPixmap();
    void drawSliceOverlay(cv::Mat& img8);

    // Helpers
    bool isAcceptableUrl(const QUrl& url) const;
    bool isAcceptablePath(const QString& path) const;
    void showDragHint();
    void clearDragHint();

    // Context-menu helpers (implemented in .cpp)
    void onSavePNG();
    void onSaveDICOM();
    bool m_refreshing = false;   // guard refreshPixmap re-entrancy

private:
    QLabel*         m_label = nullptr;
    QDockWidget*    m_metaDock = nullptr;
    QPlainTextEdit* m_metaText = nullptr;
    QSlider*        m_sliceSlider = nullptr;

    cv::Mat         m_img8;
    bool            m_hasImage = false;

    // Busy nesting counter for beginBusy/endBusy
    int             m_busyNesting = 0;

    // Accepted extensions (lowercase, no dot)
    const QStringList m_okExts{
        QStringLiteral("dcm"),
        QStringLiteral("ima"),
        QStringLiteral("h5"),
        QStringLiteral("hdf5")
    };
};
