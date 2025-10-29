#pragma once
#include <QMainWindow>
#include <QStringList>
#include <QVector>
#include <QSize>
#include <QImage>
#include <opencv2/core.hpp>


class QLabel;
class QDockWidget;
class QPlainTextEdit;
class QSlider;
class QEvent;
class QDragEnterEvent;
class QDragMoveEvent;
class QDropEvent;
class QContextMenuEvent;
class QResizeEvent;
class QUrl;
class QVBoxLayout;
class QMenu;
class QAction;

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;


    void setMetadata(const QStringList& lines);
    void appendMetadataLine(const QString& line);
    void beginNewImageCycle();
    void setImage(const cv::Mat& img8u);
    void beginBusy(const QString& message);
    void endBusy();


    void enableSliceSlider(int nSlices);
    void enableSliceSlider(bool enabled) { enableSliceSlider(enabled ? 2 : 0); }
    void enableSliceSlider(bool enabled, int maxIndex) { enableSliceSlider(enabled ? (maxIndex + 1) : 0); }
    void enableSliceSlider(bool enabled, int maxIndex, int currentIdx) {
        enableSliceSlider(enabled ? (maxIndex + 1) : 0);
        if (enabled) setSliceIndex(currentIdx);
    }
    void setSliceIndex(int idx);
    void setImageCV8U(const cv::Mat& m);

signals:

    void requestSavePNG(const QString& path);
    void requestSaveDICOM(const QString& path);


    void requestSaveDICOMSeriesOneFile(const QString& path, int rows, int cols,
                                       const QVector<QByteArray>& frames);



    void requestSaveDICOMSeriesMR(const QString& basePath,
                                  double px, double py,
                                  double sliceThickness,
                                  double spacingBetween,
                                  const QVector<double>& iop6,
                                  const QVector<double>& ipp0);


    void requestApplyNegative();
    void sliceChanged(int index);
    void fileDropped(const QString& path);
    void startOverRequested();


    void requestHistogramUpdate(const QSize& canvasSize);

public slots:
    void onNegativeModeChanged(bool on);


    void setHistogramImage(const QImage& img, const QString& tooltip);

private slots:
    void onSliderValueChanged(int v);
    void onSavePNG();
    void onSaveBatch();

protected:
    bool eventFilter(QObject* obj, QEvent* ev) override;
    void contextMenuEvent(QContextMenuEvent* ev) override;
    void resizeEvent(QResizeEvent* ev) override;


    void dragEnterEvent(QDragEnterEvent* ev) override;
    void dragMoveEvent(QDragMoveEvent* ev) override;
    void dropEvent(QDropEvent* ev) override;

private:

    void buildUi();
    QWidget* createCentralArea();
    void createMetadataDock();
    void createHistogramDock();
    void setInitialSize();
    void initRightDocksIfNeeded();


    void refreshPixmap();


    bool beginRefreshGuard();
    void endRefreshGuard();
    bool hasDrawableImage() const;
    bool labelTooSmall() const;
    bool isMultiSliceActive() const;
    cv::Mat buildDisplayImageWithOverlay();
    QImage  toQImageOwned(const cv::Mat& m) const;
    QImage  scaleForLabel(const QImage& qi) const;
    void    setPixmapAndLog(const QImage& scaled);

    void drawSliceOverlay(cv::Mat& img8);


    bool   validateImageInput(const cv::Mat& img) const;
    cv::Mat to8uMono(const cv::Mat& src) const;
    void   logMatStats(const cv::Mat& m) const;
    void   storeImage(const cv::Mat& m);
    void   updateMetadataForImage(const cv::Mat& m);
    void   repaintOnce();


    bool isAcceptableUrl(const QUrl& url) const;
    bool isAcceptablePath(const QString& path) const;
    void showDragHint();
    void clearDragHint();


    struct CtxMenuActions {
        QAction* saveSlice = nullptr;
        QAction* saveBatch = nullptr;
        QAction* negative  = nullptr;
        QAction* startOver = nullptr;
        QAction* about     = nullptr;

    };
    bool   hasImageForMenu() const;
    bool   hasMultiSlicesForMenu() const;
    QMenu* buildContextMenu(bool hasImg, CtxMenuActions& out);
    void   populateMenuForNoImage(QMenu& menu, CtxMenuActions& out);
    void   populateMenuForImage(QMenu& menu, CtxMenuActions& out,
                              bool hasMulti, bool hasImg);
    void   applyContextSelection(QAction* chosen, const CtxMenuActions& acts);


    enum class SaveFmt { PNG, DICOM, DICOM_SERIES };


    bool promptSingleSave(QString* outPath, SaveFmt* outFmt);
    void emitSingleSave(const QString& path, SaveFmt fmt);


    bool canBatchSave() const;
    bool promptBatchDestination(QString* outBasePath, SaveFmt* outFmt);
    int  computeIndexPadding(int slices) const;
    void saveBatchPNGSlices(const QString& basePath, int S, int pad);
    bool promptDicomSeriesGeometry(double* px, double* py, double* sth, double* sbs,
                                   QVector<double>* iop6, QVector<double>* ipp0);
    void emitDicomSeries(const QString& basePath, double px, double py, double sth, double sbs,
                         const QVector<double>& iop6, const QVector<double>& ipp0);


    QLabel*         m_label        = nullptr;
    QSlider*        m_sliceSlider  = nullptr;


    QDockWidget*    m_metaDock     = nullptr;
    QPlainTextEdit* m_metaText     = nullptr;


    QDockWidget*    m_histDock     = nullptr;
    QLabel*         m_histLabel    = nullptr;

    cv::Mat         m_img8;
    bool            m_hasImage     = false;

    int             m_busyNesting  = 0;
    bool            m_refreshing   = false;
    bool            m_negativeMode = false;

    const QStringList m_okExts{
        QStringLiteral("dcm"),
        QStringLiteral("ima"),
        QStringLiteral("h5"),
        QStringLiteral("hdf5")
    };


    void showAboutDialog();
    void addAboutDescription(QVBoxLayout* layout);
};
