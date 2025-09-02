#pragma once
#include <QMainWindow>
#include <QLabel>
#include <QMenu>
#include <QResizeEvent>
#include <opencv2/core.hpp>

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent=nullptr);
    void setImage(const cv::Mat& img8);

protected:
    void contextMenuEvent(QContextMenuEvent* ev) override;
    void resizeEvent(QResizeEvent* ev) override;

private slots:
    void onSavePNG();
    void onSaveDICOM();

private:
    void refreshPixmap();

    QLabel* m_label = nullptr;
    cv::Mat m_img8; // grayscale 8-bit
    cv::Mat lastImg8u_;

};
