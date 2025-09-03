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

    // Image display
    void setImage(const cv::Mat& img8);

    // --- Metadata panel API ---
    void setMetadata(const QStringList& lines); // replace all metadata
    void setMetadataText(const QString& txt);   // replace with raw text
    void appendMetadataLine(const QString& line); // append one line
    void toggleMetadataPanel();                 // show/hide the dock

protected:
    void contextMenuEvent(QContextMenuEvent* ev) override;
    void resizeEvent(QResizeEvent* ev) override;

private slots:
    void onSavePNG();
    void onSaveDICOM();

private:
    void refreshPixmap();

    // Image state
    QLabel* m_label = nullptr;
    cv::Mat m_img8;     // grayscale 8-bit
    cv::Mat lastImg8u_; // unchanged, kept for your flow

    // --- Metadata dock state ---
    QDockWidget*    m_metaDock = nullptr;
    QPlainTextEdit* m_metaText = nullptr;
};
