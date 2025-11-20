#pragma once

#include <memory>
#include <vector>

#include <QString>
#include <QStringList>
#include <QVector>
#include <QByteArray>
#include <QSize>
#include <QImage>

#include <opencv2/core.hpp>
#include <QPointer>


class MainWindow;
struct DicomDll;
class ProgressSplash;


#include "../model/io.hpp"

class AppController {
public:
    explicit AppController(MainWindow* view);
    ~AppController();


    void saveDICOMBatch(const QString& outPath);
    void saveDICOMSeriesMR(const QString& basePath,
                           double px, double py,
                           double sliceThickness,
                           double spacingBetween,
                           const QVector<double>& iop6,
                           const QVector<double>& ipp0);

    void load(const QString& pathQ);
    void show();
    void savePNG(const QString& outPath);
    void saveDICOM(const QString& outPath);


    void saveDICOMMultiframe(const QString& outPath, int rows, int cols,
                             const QVector<QByteArray>& frames);


    void onSliceChanged(int idx);
    void onStartOverRequested();
    void applyNegative();
    void toggleNegative();


    void postSplashUpdateFromEngineThread(int pct, const QString& stage);


    void onHistogramUpdateRequested(const QSize& canvas);
    bool loadDicom(const QString& pathUtf8);
private:
    struct BusyScope {
        explicit BusyScope(MainWindow* v, const QString& message);
        ~BusyScope();
        MainWindow* v_ = nullptr;
    };

    void initCore();                    // DCMTK + engine init + logging
    bool initEngine();                  // one-time MRI engine init (uses std::once_flag)
    void initViewConnections();         // all QObject::connect wiring
    void initSliceNavigationShortcuts();// keyboard shortcuts for slice navigation


    bool m_engineReady = false;

    void clearLoadState();

    bool runEngineReconstruction(const QString& pathQ,
                                 bool fftshift,
                                 std::vector<float>& host,
                                 int& S, int& H, int& W);
    void appendIsmrmrdMetadata(const QString& pathQ);

    bool failRecon(const QString& msg);
    bool succeedRecon();
    bool reconstructAllSlicesFromLib(const QString& pathQ, bool fftshift);



    void prepare_fallback();


    void doShowNow();
    void show_metadata_and_image(const QStringList& meta, const cv::Mat& u8);
    void show_gradient_with_meta(const QStringList& meta);
    void showSlices(const std::vector<cv::Mat>& frames);
    void showSlice(int idx);
    void adoptReconStackF32(const std::vector<float>& stack, int S, int H, int W);


    QImage renderHistogram(const cv::Mat& u8, const QSize& canvas, bool negativeMode, QString* tooltip);


    static cv::Mat to_u8(const cv::Mat& f32);
    static cv::Mat vecf32_to_u8(const std::vector<float>& v, int H, int W);
    static cv::Mat make_gradient(int H, int W);
    bool m_negativeMode = false;


    std::vector<cv::Mat> m_slices8_base;
    cv::Mat              m_display8_base;


    static cv::Mat invert8u(const cv::Mat& src);
    void captureNegativeBaseIfNeeded();
    void closeSplashIfAny();

    QStringList formatDicomMeta(const io::DicomMeta& m) const;

private:

    MainWindow* m_view = nullptr;


    io::ProbeResult m_probe;
    QStringList     m_meta;


    std::unique_ptr<DicomDll> m_dicom;


    cv::Mat              m_display8;
    cv::Mat              m_lastImg8;
    std::vector<cv::Mat> m_slices8;
    int                  m_currentSlice = 0;


    QString              m_sourcePathQ;


    QPointer<ProgressSplash> m_splash;


    bool decodeDicomToFrames8(const std::string& path,
                              std::vector<cv::Mat>& outFrames8,
                              std::string& why);


    static cv::Mat convert16To8(const cv::Mat& f16);


    void adoptFrames8ToState(const std::vector<cv::Mat>& frames8);


    void scheduleOrDoDicomMetadataRead(const std::string& path);


};
