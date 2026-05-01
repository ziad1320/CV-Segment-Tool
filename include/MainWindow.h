#pragma once

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QTextEdit>    // NEW: For the log terminal
#include <QGroupBox>    // NEW: For the boxes
#include <opencv2/opencv.hpp>

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow() = default;

private slots:
    void openImage();
    void processThreshold();   // Separate slot for thresholding
    void processSegmentation(); // Separate slot for segmentation

private:
    // UI Elements - Images
    QLabel *originalImageLabel;
    QLabel *processedImageLabel;

    // UI Elements - Sidebar
    QPushButton *loadButton;
    QComboBox *thresholdSelect;
    QComboBox *segmentationSelect;
    QPushButton *applyThresholdBtn;
    QPushButton *applySegmentationBtn;
    QTextEdit *logTerminal; // Our internal terminal

    // Data
    cv::Mat currentImage;

    // Helper functions
    void log(const QString &message);
    QImage cvMatToQImage(const cv::Mat &inMat);
    void displayResult(const cv::Mat &img);
};