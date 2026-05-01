#include "../include/MainWindow.h"
#include "../include/Thresholding.h"
#include "../include/Segmentation.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QWidget>
#include <QDebug>

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
    
    // The unified algorithm dropdown
    algorithmSelect = new QComboBox(this);
    algorithmSelect->addItem("Otsu Thresholding");
    algorithmSelect->addItem("Optimal Thresholding");
    algorithmSelect->addItem("Local Thresholding");
    algorithmSelect->addItem("Spectral Thresholding");
    // --- NEW SEGMENTATION ITEMS ---
    algorithmSelect->insertSeparator(algorithmSelect->count()); // Visual line in dropdown
    algorithmSelect->addItem("K-Means Segmentation");
    algorithmSelect->addItem("Mean Shift Segmentation");

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
        // Using toLocal8Bit to safely handle Windows paths
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
    QString selectedAlgorithm = algorithmSelect->currentText();

    // Route the image to the correct algorithm in Thresholding.cpp
    if (selectedAlgorithm == "Otsu Thresholding") {
        processedMat = Thresholding::applyOtsu(currentImage);
    } 
    else if (selectedAlgorithm == "Optimal Thresholding") {
        processedMat = Thresholding::applyOptimal(currentImage);
    }
    else if (selectedAlgorithm == "Local Thresholding") {
        processedMat = Thresholding::applyLocal(currentImage);
    }
    else if (selectedAlgorithm == "Spectral Thresholding") {
        processedMat = Thresholding::applySpectral(currentImage);
    }
    // --- NEW SEGMENTATION ROUTES ---
    else if (selectedAlgorithm == "K-Means Segmentation") {
        // We use the default k=4, but later you could add a UI slider to let the user change 'k'!
        processedMat = Segmentation::applyKMeans(currentImage, 4); 
    }
    else if (selectedAlgorithm == "Mean Shift Segmentation") {
        processedMat = Segmentation::applyMeanShift(currentImage);
    }

    // Display the result
    if (!processedMat.empty()) {
        QImage qimg = cvMatToQImage(processedMat);
        processedImageLabel->setPixmap(QPixmap::fromImage(qimg).scaled(processedImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
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