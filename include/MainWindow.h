#pragma once

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <opencv2/opencv.hpp>

class MainWindow : public QMainWindow {
    Q_OBJECT // Mandatory macro for Qt signals and slots

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow() = default;

private slots:
    void openImage();
    void processImage();

private:
    // UI Elements
    QLabel *originalImageLabel;
    QLabel *processedImageLabel;
    QPushButton *loadButton;
    QPushButton *processButton;
    QComboBox *algorithmSelect;

    // Data
    cv::Mat currentImage;

    // The Bridge Function
    QImage cvMatToQImage(const cv::Mat &inMat);
};