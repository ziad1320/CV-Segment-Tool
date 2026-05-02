#include "../include/MainWindow.h"
#include "../include/Thresholding.h"
#include "../include/Segmentation.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QDateTime>
#include <QMouseEvent>
#include <QPainter>
#include <QApplication>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setWindowTitle("CVSegmenter Pro - Interactive");
    resize(1200, 750); // Slightly taller to fit all 3 boxes

    // --- 1. IMAGE AREA (Top Right) ---
    originalImageLabel = new QLabel("Original Image\n(Click to select K-Means seeds)", this);
    originalImageLabel->setAlignment(Qt::AlignCenter);
    originalImageLabel->setStyleSheet("border: 2px solid #444; background-color: #222; color: gray;");
    originalImageLabel->setMinimumSize(400, 300);
    originalImageLabel->installEventFilter(this); 

    processedImageLabel = new QLabel("Processed Result", this);
    processedImageLabel->setAlignment(Qt::AlignCenter);
    processedImageLabel->setStyleSheet("border: 2px solid #444; background-color: #222; color: gray;");
    processedImageLabel->setMinimumSize(400, 300);

    QHBoxLayout *imageLayout = new QHBoxLayout();
    imageLayout->addWidget(originalImageLabel, 1);
    imageLayout->addWidget(processedImageLabel, 1);

    // --- 2. TERMINAL AREA (Bottom Right) ---
    logTerminal = new QTextEdit(this);
    logTerminal->setReadOnly(true);
    logTerminal->setStyleSheet("background-color: black; color: #00FF00; font-family: 'Consolas'; font-size: 10pt;");
    logTerminal->setFixedHeight(300); // Keep it as a fixed bottom panel

    // --- 3. ASSEMBLE RIGHT SIDE (Images + Terminal) ---
    QWidget *rightArea = new QWidget(this);
    QVBoxLayout *rightLayout = new QVBoxLayout(rightArea);
    rightLayout->addLayout(imageLayout); // Images go on top
    rightLayout->addWidget(new QLabel("Output Terminal:", this));
    rightLayout->addWidget(logTerminal); // Terminal goes on bottom

    // --- 4. SIDEBAR AREA (Left Side) ---
    QWidget *sidebar = new QWidget(this);
    sidebar->setFixedWidth(320);
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

    // Interactive K-Means Box
    QGroupBox *kmeansBox = new QGroupBox("K-Means", this);
    kmeansBox->setStyleSheet("QGroupBox { border: 1px solid #00AAFF; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #00AAFF; }");
    QVBoxLayout *kLayout = new QVBoxLayout(kmeansBox);
    
    QHBoxLayout *kRow = new QHBoxLayout();
    kRow->addWidget(new QLabel("Clusters (K):", this));
    kSpinner = new QSpinBox(this);
    kSpinner->setRange(2, 20);
    kSpinner->setValue(4);
    kRow->addWidget(kSpinner);
    kLayout->addLayout(kRow);

    QHBoxLayout *iterRow = new QHBoxLayout();
    iterRow->addWidget(new QLabel("Max Iterations:", this));
    iterSpinner = new QSpinBox(this);
    iterSpinner->setRange(1, 200);
    iterSpinner->setValue(50);
    iterRow->addWidget(iterSpinner);
    kLayout->addLayout(iterRow);

    runKMeansBtn = new QPushButton("Run K-Means", this);
    clearSeedsBtn = new QPushButton("Clear Clicked Seeds", this);
    kLayout->addWidget(runKMeansBtn);
    kLayout->addWidget(clearSeedsBtn);
    sideLayout->addWidget(kmeansBox);

    // Standard Segmentation Box
    QGroupBox *segBox = new QGroupBox("Other Segmentation Algorithms", this);
    QVBoxLayout *sLayout = new QVBoxLayout(segBox);
    segmentationSelect = new QComboBox(this);
    segmentationSelect->addItems({"Mean Shift Segmentation", "Region Growing", "Agglomerative Clustering"});
    applySegmentationBtn = new QPushButton("Apply Selected", this);
    sLayout->addWidget(segmentationSelect);
    sLayout->addWidget(applySegmentationBtn);
    sideLayout->addWidget(segBox);
    
    // Add a stretching space at the bottom of the sidebar so the boxes don't space out weirdly
    sideLayout->addStretch();

    // --- 5. MAIN LAYOUT ASSEMBLY ---
    QWidget *centralWidget = new QWidget(this);
    QHBoxLayout *mainLayout = new QHBoxLayout(centralWidget);
    mainLayout->addWidget(sidebar, 1);     // Sidebar on left
    mainLayout->addWidget(rightArea, 4);   // Right Area (Images + Terminal) on right
    setCentralWidget(centralWidget);

    // --- 6. CONNECTIONS ---
    connect(loadButton, &QPushButton::clicked, this, &MainWindow::openImage);
    connect(applyThresholdBtn, &QPushButton::clicked, this, &MainWindow::processThreshold);
    connect(applySegmentationBtn, &QPushButton::clicked, this, &MainWindow::processSegmentation);
    connect(runKMeansBtn, &QPushButton::clicked, this, &MainWindow::runInteractiveKMeans);
    connect(clearSeedsBtn, &QPushButton::clicked, this, &MainWindow::clearKMeansSeeds);

    log("Ready. Load an image, click on colors you want to segment, and hit Run K-Means!");
}

// ---------------------------------------------------------
// Qt6 Mouse Event Interception
// ---------------------------------------------------------
bool MainWindow::eventFilter(QObject *obj, QEvent *event) {
    if (obj == originalImageLabel && event->type() == QEvent::MouseButtonPress) {
        if (currentImage.empty()) return false;

        QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
        
        QPixmap pixmap = originalImageLabel->pixmap(); 
        if (pixmap.isNull()) return false;

        int labelW = originalImageLabel->width();
        int labelH = originalImageLabel->height();
        int pixmapW = pixmap.width();
        int pixmapH = pixmap.height();

        // int xOffset = (labelW - pixmapW) / 2;
        // int yOffset = (labelH - pixmapH) / 2;

        int clickX = mouseEvent->position().toPoint().x();
        int clickY = mouseEvent->position().toPoint().y();

        if (clickX >= 0 && clickX < pixmapW && clickY >= 0 && clickY < pixmapH) {
            int trueX = (clickX * currentImage.cols) / pixmapW;
            int trueY = (clickY * currentImage.rows) / pixmapH;

            cv::Vec3b color = currentImage.at<cv::Vec3b>(trueY, trueX);
            userSeeds.push_back(color);

            log(QString("Picked Seed Color: [B:%1, G:%2, R:%3]").arg(color[0]).arg(color[1]).arg(color[2]));
            drawSeedOnImage(QPoint(clickX, clickY));
            
            if (userSeeds.size() > (size_t)kSpinner->value()) {
                kSpinner->setValue(userSeeds.size());
            }
        }
        return true;
    }
    return QMainWindow::eventFilter(obj, event);
}

void MainWindow::drawSeedOnImage(QPoint pos) {
    QPixmap pm = originalImageLabel->pixmap(); // FIXED for Qt6
    QPainter painter(&pm);
    painter.setPen(QPen(Qt::red, 3));
    painter.drawLine(pos.x() - 5, pos.y(), pos.x() + 5, pos.y());
    painter.drawLine(pos.x(), pos.y() - 5, pos.x(), pos.y() + 5);
    originalImageLabel->setPixmap(pm);
}

void MainWindow::clearKMeansSeeds() {
    userSeeds.clear();
    if (!currentImage.empty()) {
        QImage qimg = cvMatToQImage(currentImage);
        originalImageLabel->setPixmap(QPixmap::fromImage(qimg).scaled(originalImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    log("Cleared all interactive seeds.");
}

void MainWindow::runInteractiveKMeans() {
    if (currentImage.empty()) { 
        log("Warning: Load an image first!"); 
        return; 
    }
    
    int k = kSpinner->value();
    int iters = iterSpinner->value();
    log(QString("Running K-Means (K=%1, Iters=%2). Calculating...").arg(k).arg(iters));
    
    QApplication::setOverrideCursor(Qt::WaitCursor);
    
    QString algorithmLog = "";
    cv::Mat result = Segmentation::applyKMeans(currentImage, k, iters, userSeeds, algorithmLog);
    
    QApplication::restoreOverrideCursor();

    if (!algorithmLog.isEmpty()) log("   ↳ " + algorithmLog);
    displayResult(result);
}

// ---------------------------------------------------------
// General Process Routing
// ---------------------------------------------------------

void MainWindow::log(const QString &message) {
    QString time = QDateTime::currentDateTime().toString("hh:mm:ss");
    logTerminal->append(QString("[%1] %2").arg(time, message));
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
        clearKMeansSeeds(); 
        
        log("Loaded image: " + fileName.section('/', -1));
    }
}

void MainWindow::processThreshold() {
    if (currentImage.empty()) { 
        log("Warning: Load an image first!"); 
        return; 
    }
    
    cv::Mat grayImage;
    cv::Mat processedMat;
    QString selectedAlgorithm = thresholdSelect->currentText();
    log("Running " + selectedAlgorithm + "...");

    QString algorithmLog = ""; 

    if (selectedAlgorithm == "Otsu Thresholding") {
        processedMat = Thresholding::applyOtsu(grayImage, algorithmLog);
    } 
    else if (selectedAlgorithm == "Optimal Thresholding") {
        processedMat = Thresholding::applyOptimal(grayImage, algorithmLog);
    }
    else if (selectedAlgorithm == "Local Thresholding") {
        processedMat = Thresholding::applyLocal(grayImage);
    }
    else if (selectedAlgorithm == "Spectral Thresholding") {
        processedMat = Thresholding::applySpectral(grayImage);
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

    QApplication::setOverrideCursor(Qt::WaitCursor);

    if (selectedAlgorithm == "Mean Shift Segmentation") {
        processedMat = Segmentation::applyMeanShift(currentImage);
    }
    else if (selectedAlgorithm == "Region Growing") {  
        processedMat = Segmentation::applyRegionGrowing(currentImage);
    }
    else if (selectedAlgorithm == "Agglomerative Clustering") { 
        processedMat = Segmentation::applyAgglomerative(currentImage);
    }

    QApplication::restoreOverrideCursor();

    displayResult(processedMat);
    log("Finished " + selectedAlgorithm);
}

void MainWindow::displayResult(const cv::Mat &img) {
    if (!img.empty()) {
        QImage qimg = cvMatToQImage(img);
        processedImageLabel->setPixmap(QPixmap::fromImage(qimg).scaled(processedImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

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
        case CV_8UC4: {
            QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_RGBA8888);
            return image.copy();
        }
        default:
            return QImage();
    }
}