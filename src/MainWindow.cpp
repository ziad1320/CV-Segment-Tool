#include "../include/MainWindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QWidget>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setWindowTitle("CVSegmenter - Project 4");
    resize(900, 500);

    // 1. Create UI Elements
    originalImageLabel = new QLabel("Original Image", this);
    originalImageLabel->setAlignment(Qt::AlignCenter);
    originalImageLabel->setStyleSheet("border: 1px solid black;");
    originalImageLabel->setMinimumSize(400, 300);

    processedImageLabel = new QLabel("Processed Image", this);
    processedImageLabel->setAlignment(Qt::AlignCenter);
    processedImageLabel->setStyleSheet("border: 1px solid black;");
    processedImageLabel->setMinimumSize(400, 300);

    loadButton = new QPushButton("Load Image", this);
    processButton = new QPushButton("Apply Algorithm", this);
    
    algorithmSelect = new QComboBox(this);
    algorithmSelect->addItem("Otsu Thresholding");
    // We will add the other 7 algorithms here later

    // 2. Arrange UI with Layouts
    QWidget *centralWidget = new QWidget(this);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
    
    QHBoxLayout *imageLayout = new QHBoxLayout();
    imageLayout->addWidget(originalImageLabel);
    imageLayout->addWidget(processedImageLabel);

    QHBoxLayout *controlLayout = new QHBoxLayout();
    controlLayout->addWidget(loadButton);
    controlLayout->addWidget(algorithmSelect);
    controlLayout->addWidget(processButton);

    mainLayout->addLayout(imageLayout);
    mainLayout->addLayout(controlLayout);
    setCentralWidget(centralWidget);

    // 3. Connect Buttons to Functions
    connect(loadButton, &QPushButton::clicked, this, &MainWindow::openImage);
    connect(processButton, &QPushButton::clicked, this, &MainWindow::processImage);
}

void MainWindow::openImage() {
    QString fileName = QFileDialog::getOpenFileName(this, "Open Image", "", "Images (*.png *.xpm *.jpg *.bmp)");
    if (!fileName.isEmpty()) {
        qDebug() << "Attempting to load file from path:" << fileName;
        currentImage = cv::imread(fileName.toLocal8Bit().constData());
        if (currentImage.empty()) {
            QMessageBox::warning(this, "Error", "Failed to load image!");
            return;
        }
        
        QImage qimg = cvMatToQImage(currentImage);
        originalImageLabel->setPixmap(QPixmap::fromImage(qimg).scaled(originalImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        processedImageLabel->clear();
        processedImageLabel->setText("Ready for processing");
    }
}

void MainWindow::processImage() {
    if (currentImage.empty()) return;

    cv::Mat processedMat;

    if (algorithmSelect->currentText() == "Otsu Thresholding") {
        cv::Mat gray;
        if (currentImage.channels() == 3) {
            cv::cvtColor(currentImage, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = currentImage.clone();
        }
        
        // --- NEW DEBUGGING CODE ---
        // 1. Capture the threshold value Otsu calculated
        double otsuThresh = cv::threshold(gray, processedMat, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        
        // 2. Print it to the console
        qDebug() << "======================================";
        qDebug() << "Calculated Otsu Threshold Value:" << otsuThresh;
        qDebug() << "======================================";

        // --- THE PIXEL COUNTER ---
        int whitePixels = cv::countNonZero(processedMat);
        int totalPixels = processedMat.rows * processedMat.cols;
        qDebug() << "Total Pixels :" << totalPixels;
        qDebug() << "White Pixels :" << whitePixels;
        qDebug() << "Black Pixels :" << (totalPixels - whitePixels);
        
        // cv::imwrite("debug_otsu_result.png", processedMat);

        // // 3. Save the image directly to your build folder to bypass Qt's UI
        // cv::imwrite("debug_otsu_result.png", processedMat);
    }

    // Display the result in Qt
    QImage qimg = cvMatToQImage(processedMat);
    processedImageLabel->setPixmap(QPixmap::fromImage(qimg).scaled(processedImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

// THE BRIDGE: Converts OpenCV Mat to Qt Image
QImage MainWindow::cvMatToQImage(const cv::Mat &inMat) {
    switch (inMat.type()) {
        // 8-bit, 3 channel (Standard Color)
        case CV_8UC3: {
            QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_RGB888);
            return image.rgbSwapped(); // OpenCV uses BGR, Qt uses RGB
        }
        // 8-bit, 1 channel (Grayscale/Thresholded)
        case CV_8UC1: {
            QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_Grayscale8);
            return image.copy(); // Deep copy needed for grayscale
        }
        default:
            qWarning() << "CVSegmenter: Image format not supported by bridge!";
            return QImage();
    }
}