#include "../include/MainWindow.h"
#include "../include/Thresholding.h"
#include "../include/Segmentation.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QWidget>
#include <QDateTime>
#include <QGroupBox>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setWindowTitle("CVSegmenter Pro - Project 4");
    resize(1200, 700);

    // --- 1. IMAGE AREA (Right Side Now) ---
    originalImageLabel = new QLabel("Original Image", this);
    originalImageLabel->setAlignment(Qt::AlignCenter);
    originalImageLabel->setStyleSheet("border: 2px solid #444; background-color: #222; color: gray;");
    originalImageLabel->setMinimumSize(400, 300);

    processedImageLabel = new QLabel("Processed Result", this);
    processedImageLabel->setAlignment(Qt::AlignCenter);
    processedImageLabel->setStyleSheet("border: 2px solid #444; background-color: #222; color: gray;");
    processedImageLabel->setMinimumSize(400, 300);

    QHBoxLayout *imageLayout = new QHBoxLayout();
    imageLayout->addWidget(originalImageLabel, 1);
    imageLayout->addWidget(processedImageLabel, 1);

    // --- 2. SIDEBAR AREA (Left Side Now) ---
    QWidget *sidebar = new QWidget(this);
    sidebar->setFixedWidth(300);
    QVBoxLayout *sideLayout = new QVBoxLayout(sidebar);

    // General Controls
    loadButton = new QPushButton("Load New Image", this);
    loadButton->setMinimumHeight(40);
    sideLayout->addWidget(loadButton);

    // Thresholding Box
    QGroupBox *threshBox = new QGroupBox("Thresholding Algorithms", this);
    QVBoxLayout *tLayout = new QVBoxLayout(threshBox);
    thresholdSelect = new QComboBox(this);
    thresholdSelect->addItems({"Otsu Thresholding", "Optimal Thresholding", "Local Thresholding", "Spectral Thresholding"});
    applyThresholdBtn = new QPushButton("Apply Threshold", this);
    tLayout->addWidget(thresholdSelect);
    tLayout->addWidget(applyThresholdBtn);
    sideLayout->addWidget(threshBox);

    // Segmentation Box
    QGroupBox *segBox = new QGroupBox("Segmentation Algorithms", this);
    QVBoxLayout *sLayout = new QVBoxLayout(segBox);
    segmentationSelect = new QComboBox(this);
    segmentationSelect->addItems({"K-Means Segmentation", "Mean Shift Segmentation", "Region Growing", "Agglomerative Clustering"});
    applySegmentationBtn = new QPushButton("Apply Segmentation", this);
    sLayout->addWidget(segmentationSelect);
    sLayout->addWidget(applySegmentationBtn);
    sideLayout->addWidget(segBox);

    // Output Terminal
    logTerminal = new QTextEdit(this);
    logTerminal->setReadOnly(true);
    logTerminal->setStyleSheet("background-color: black; color: #00FF00; font-family: 'Consolas'; font-size: 10pt;");
    sideLayout->addWidget(new QLabel("Output Terminal:", this));
    sideLayout->addWidget(logTerminal);

    // --- 3. MAIN LAYOUT ASSEMBLY ---
    QWidget *centralWidget = new QWidget(this);
    QHBoxLayout *mainLayout = new QHBoxLayout(centralWidget);
    
    // SWAPPED ORDER: Sidebar is added first (Left), Images second (Right)
    mainLayout->addWidget(sidebar, 1);     
    mainLayout->addLayout(imageLayout, 4); 
    
    setCentralWidget(centralWidget);

    // --- 4. CONNECTIONS ---
    connect(loadButton, &QPushButton::clicked, this, &MainWindow::openImage);
    connect(applyThresholdBtn, &QPushButton::clicked, this, &MainWindow::processThreshold);
    connect(applySegmentationBtn, &QPushButton::clicked, this, &MainWindow::processSegmentation);

    log("Application Started. Ready to load image...");
}

// Custom function to print timestamps and text to our UI terminal
void MainWindow::log(const QString &message) {
    logTerminal->append(QString("%1").arg( message));
}

void MainWindow::openImage() {
    QString fileName = QFileDialog::getOpenFileName(this, "Open Image", "", "Images (*.png *.xpm *.jpg *.bmp)");
    if (!fileName.isEmpty()) {
        currentImage = cv::imread(fileName.toLocal8Bit().constData());
        if (currentImage.empty()) {
            QMessageBox::warning(this, "Error", "Failed to load image!");
            log("ERROR: Failed to load image.");
            return;
        }
        
        QImage qimg = cvMatToQImage(currentImage);
        originalImageLabel->setPixmap(QPixmap::fromImage(qimg).scaled(originalImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        processedImageLabel->clear();
        processedImageLabel->setText("Algorithm Result Area");
        
        log("Loaded image: " + fileName.section('/', -1));
    }
}

void MainWindow::processThreshold() {
    if (currentImage.empty()) { 
        log("Warning: Load an image first!"); 
        return; 
    }
    
    cv::Mat processedMat;
    QString selectedAlgorithm = thresholdSelect->currentText();
    log("Running " + selectedAlgorithm + "...");

    //empty string to catch messages from the backend
    QString algorithmLog = "";

    if (selectedAlgorithm == "Otsu Thresholding") {
        processedMat = Thresholding::applyOtsu(currentImage, algorithmLog);
    } 
    else if (selectedAlgorithm == "Optimal Thresholding") {
        processedMat = Thresholding::applyOptimal(currentImage, algorithmLog);
    }
    else if (selectedAlgorithm == "Local Thresholding") {
        processedMat = Thresholding::applyLocal(currentImage);
    }
    else if (selectedAlgorithm == "Spectral Thresholding") {
        processedMat = Thresholding::applySpectral(currentImage);
    }

    if (!algorithmLog.isEmpty()) {
        log("   ↳ " + algorithmLog);
    }

    displayResult(processedMat);
    log("Finished " + selectedAlgorithm);
}

void MainWindow::processSegmentation() {
    if (currentImage.empty()) { 
        log("Warning: Load an image first!"); 
        return; 
    }
    
    cv::Mat processedMat;
    QString selectedAlgorithm = segmentationSelect->currentText();
    log("Running " + selectedAlgorithm + " (Please wait)...");

    if (selectedAlgorithm == "K-Means Segmentation") {
        processedMat = Segmentation::applyKMeans(currentImage, 4); 
    }
    else if (selectedAlgorithm == "Mean Shift Segmentation") {
        processedMat = Segmentation::applyMeanShift(currentImage);
    }
    else if (selectedAlgorithm == "Region Growing") {  
        processedMat = Segmentation::applyRegionGrowing(currentImage);
    }
    else if (selectedAlgorithm == "Agglomerative Clustering") { 
        processedMat = Segmentation::applyAgglomerative(currentImage);
    }

    displayResult(processedMat);
    log("Finished " + selectedAlgorithm);
}

void MainWindow::displayResult(const cv::Mat &img) {
    if (!img.empty()) {
        QImage qimg = cvMatToQImage(img);
        processedImageLabel->setPixmap(QPixmap::fromImage(qimg).scaled(processedImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

// THE BRIDGE: Converts OpenCV Mat to Qt Image
QImage MainWindow::cvMatToQImage(const cv::Mat &inMat) {
    switch (inMat.type()) {
        case CV_8UC3: {
            QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_RGB888);
            return image.rgbSwapped(); 
        }
        case CV_8UC1: {
            QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_Grayscale8);
            return image.copy(); 
        }
        default:
            qWarning() << "CVSegmenter: Image format not supported by bridge!";
            return QImage();
    }
}