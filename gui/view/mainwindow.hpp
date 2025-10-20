#pragma once
#include <QMainWindow>
#include <QLabel>
#include <QMenu>
#include <QResizeEvent>
#include <QDockWidget>
#include <QPlainTextEdit>
#include <QStringList>
#include <QString>
#include <opencv2/core.hpp>

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent=nullptr);

    // existing API
    void setImage(const cv::Mat& img8);
    void setMetadata(const QStringList& lines);
    void appendMetadataLine(const QString& line);
    void beginNewImageCycle();

    // NEW: busy status API
    void beginBusy(const QString& message);
    void endBusy();

signals:
    void requestSavePNG(const QString& path);
    void requestSaveDICOM(const QString& path);

protected:
    void contextMenuEvent(QContextMenuEvent* ev) override;
    void resizeEvent(QResizeEvent* ev) override;

private slots:
    void onSavePNG();
    void onSaveDICOM();

private:
    void refreshPixmap();

    QLabel* m_label = nullptr;
    cv::Mat m_img8;
    bool    m_hasImage = false;

    QDockWidget*    m_metaDock = nullptr;
    QPlainTextEdit* m_metaText = nullptr;

    // NEW: track nested busy scopes (safe in reentrant paths)
    int m_busyNesting = 0;
};
