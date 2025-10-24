#pragma once
#include <QMainWindow>
#include <QStringList>
#include <QVector>
#include <QSize>
#include <QImage>
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
class QVBoxLayout;
class QMenu;
class QAction;

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

    // View API used by the controller
    void setMetadata(const QStringList& lines);
    void appendMetadataLine(const QString& line);
    void beginNewImageCycle();
    void setImage(const cv::Mat& img8u);
    void beginBusy(const QString& message);
    void endBusy();

    // Slice slider API (overloads keep AppController happy)
    void enableSliceSlider(int nSlices);                        // 0 or 1 -> disabled; >=2 -> enabled
    void enableSliceSlider(bool enabled) { enableSliceSlider(enabled ? 2 : 0); }
    void enableSliceSlider(bool enabled, int maxIndex) { enableSliceSlider(enabled ? (maxIndex + 1) : 0); }
    void enableSliceSlider(bool enabled, int maxIndex, int currentIdx) {
        enableSliceSlider(enabled ? (maxIndex + 1) : 0);
        if (enabled) setSliceIndex(currentIdx);
    }
    void setSliceIndex(int idx);
    void setImageCV8U(const cv::Mat& m);   // expects CV_8UC1; shows slice + requests histogram update

signals:
    // Existing single-slice save signals
    void requestSavePNG(const QString& path);
    void requestSaveDICOM(const QString& path);

    // Optional: multi-frame Secondary Capture (one file)
    void requestSaveDICOMSeriesOneFile(const QString& path, int rows, int cols,
                                       const QVector<QByteArray>& frames);

    // MR series (N files; one file per slice)
    // basePath like ".../series.dcm" -> controller writes series_0001.dcm, etc.
    void requestSaveDICOMSeriesMR(const QString& basePath,
                                  double px, double py,
                                  double sliceThickness,
                                  double spacingBetween,
                                  const QVector<double>& iop6,
                                  const QVector<double>& ipp0);

    // Controller listens to these:
    void requestApplyNegative();      // controller toggles then calls onNegativeModeChanged(bool)
    void sliceChanged(int index);
    void fileDropped(const QString& path);
    void startOverRequested();

    // Histogram (MVC): the view requests, the controller computes & calls back
    void requestHistogramUpdate(const QSize& canvasSize);

public slots:
    void onNegativeModeChanged(bool on);

    // Controller → View: set the rendered histogram image + tooltip
    void setHistogramImage(const QImage& img, const QString& tooltip);

private slots:
    void onSliderValueChanged(int v);  // wheel/slider -> emit sliceChanged
    void onSavePNG();                  // unified: user picks PNG or DICOM (single slice)
    void onSaveBatch();                // batch: PNG -> many files; DICOM -> MR series (many files)

protected:
    bool eventFilter(QObject* obj, QEvent* ev) override;
    void contextMenuEvent(QContextMenuEvent* ev) override;
    void resizeEvent(QResizeEvent* ev) override;

    // Drag & drop
    void dragEnterEvent(QDragEnterEvent* ev) override;
    void dragMoveEvent(QDragMoveEvent* ev) override;
    void dropEvent(QDropEvent* ev) override;

private:
    // --- Constructor decomposition ---
    void buildUi();
    QWidget* createCentralArea();   // image label + slider row
    void createMetadataDock();      // right dock, hidden by default (Image details)
    void createHistogramDock();     // grayscale histogram dock (floatable; view-only)
    void setInitialSize();          // initial window geometry
    void initRightDocksIfNeeded();  // one-time split/resize when first image arrives

    // --- Paint path (refactored) ---
    void refreshPixmap();

    // Helpers used by refreshPixmap
    bool beginRefreshGuard();          // sets m_refreshing and logs if already refreshing
    void endRefreshGuard();            // clears m_refreshing
    bool hasDrawableImage() const;     // image+label present
    bool labelTooSmall() const;        // label area too small for a render
    bool isMultiSliceActive() const;   // slider enabled and >=2 slices
    cv::Mat buildDisplayImageWithOverlay(); // m_img8 or copy with overlay
    QImage  toQImageOwned(const cv::Mat& m) const;
    QImage  scaleForLabel(const QImage& qi) const;
    void    setPixmapAndLog(const QImage& scaled);

    void drawSliceOverlay(cv::Mat& img8);   // draws "Slice i/N" on image when multi-slice

    // --- setImage refactor helpers ---
    bool   validateImageInput(const cv::Mat& img) const;
    cv::Mat to8uMono(const cv::Mat& src) const;
    void   logMatStats(const cv::Mat& m) const;
    void   storeImage(const cv::Mat& m);
    void   updateMetadataForImage(const cv::Mat& m);
    void   repaintOnce();

    // --- DnD helpers ---
    bool isAcceptableUrl(const QUrl& url) const;
    bool isAcceptablePath(const QString& path) const;
    void showDragHint();
    void clearDragHint();

    // --- Context menu decomposition ---
    struct CtxMenuActions {
        QAction* saveSlice = nullptr;
        QAction* saveBatch = nullptr;
        QAction* negative  = nullptr;
        QAction* startOver = nullptr;
        QAction* about     = nullptr;
        // "Histogram" and "Image Info" are created on the fly via objectName
    };
    bool   hasImageForMenu() const;
    bool   hasMultiSlicesForMenu() const;
    QMenu* buildContextMenu(bool hasImg, CtxMenuActions& out);          // caller owns
    void   populateMenuForNoImage(QMenu& menu, CtxMenuActions& out);
    void   populateMenuForImage(QMenu& menu, CtxMenuActions& out,
                              bool hasMulti, bool hasImg);
    void   applyContextSelection(QAction* chosen, const CtxMenuActions& acts);

    // --- Save helpers (refactor target) ---
    enum class SaveFmt { PNG, DICOM, DICOM_SERIES };

    // Single-file (slice) save helpers
    bool promptSingleSave(QString* outPath, SaveFmt* outFmt);       // dialog
    void emitSingleSave(const QString& path, SaveFmt fmt);          // emit proper signal

    // Batch save helpers
    bool canBatchSave() const;
    bool promptBatchDestination(QString* outBasePath, SaveFmt* outFmt);
    int  computeIndexPadding(int slices) const;
    void saveBatchPNGSlices(const QString& basePath, int S, int pad);
    bool promptDicomSeriesGeometry(double* px, double* py, double* sth, double* sbs,
                                   QVector<double>* iop6, QVector<double>* ipp0);
    void emitDicomSeries(const QString& basePath, double px, double py, double sth, double sbs,
                         const QVector<double>& iop6, const QVector<double>& ipp0);

    // --- Widgets/state ---
    QLabel*         m_label        = nullptr;
    QSlider*        m_sliceSlider  = nullptr;

    // Metadata dock (embedded but floatable)
    QDockWidget*    m_metaDock     = nullptr;
    QPlainTextEdit* m_metaText     = nullptr;

    // Histogram dock (embedded but floatable)
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

    // About dialog
    void showAboutDialog();
    void addAboutDescription(QVBoxLayout* layout);
};
