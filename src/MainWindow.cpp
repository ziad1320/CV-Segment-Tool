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
    resize(1200, 850); // Slightly taller to fit all the new dedicated boxes

    // --- 1. IMAGE AREA (Top Right) ---
    originalImageLabel = new QLabel("Original Image\n(Click to select Seeds)", this);
    originalImageLabel->setAlignment(Qt::AlignCenter);
    originalImageLabel->setStyleSheet("border: 2px solid #444; background-color: #222; color: gray;");
    originalImageLabel->setMinimumSize(500, 400);
    originalImageLabel->installEventFilter(this); 

    processedImageLabel = new QLabel("Processed Result", this);
    processedImageLabel->setAlignment(Qt::AlignCenter);
    processedImageLabel->setStyleSheet("border: 2px solid #444; background-color: #222; color: gray;");
    processedImageLabel->setMinimumSize(500, 400);

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

    // Dedicated Mean Shift Box
    QGroupBox *meanShiftBox = new QGroupBox("Mean Shift Segmentation", this);
    meanShiftBox->setStyleSheet("QGroupBox { border: 1px solid #FFAA00; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #FFAA00; }");
    QVBoxLayout *msLayout = new QVBoxLayout(meanShiftBox);

    QHBoxLayout *spatialRow = new QHBoxLayout();
    spatialRow->addWidget(new QLabel("Spatial Bandwidth:", this));
    spatialBandwidthSpinner = new QDoubleSpinBox(this);
    spatialBandwidthSpinner->setRange(1.0, 50.0);
    spatialBandwidthSpinner->setValue(10.0); // Default
    spatialRow->addWidget(spatialBandwidthSpinner);
    msLayout->addLayout(spatialRow);

    QHBoxLayout *colorRow = new QHBoxLayout();
    colorRow->addWidget(new QLabel("Color Bandwidth:", this));
    colorBandwidthSpinner = new QDoubleSpinBox(this);
    colorBandwidthSpinner->setRange(5.0, 100.0);
    colorBandwidthSpinner->setValue(25.0); // Default
    colorRow->addWidget(colorBandwidthSpinner);
    msLayout->addLayout(colorRow);

    QHBoxLayout *msIterRow = new QHBoxLayout();
    msIterRow->addWidget(new QLabel("Max Iters:", this));
    msIterSpinner = new QSpinBox(this);
    msIterSpinner->setRange(1, 100);
    msIterSpinner->setValue(10); // Default
    msIterRow->addWidget(msIterSpinner);
    msLayout->addLayout(msIterRow);

    runMeanShiftBtn = new QPushButton("Run Mean Shift", this);
    msLayout->addWidget(runMeanShiftBtn);
    sideLayout->addWidget(meanShiftBox);

    // --- NEW: Dedicated Region Growing Box ---
    QGroupBox *rgBox = new QGroupBox("Region Growing", this);
    rgBox->setStyleSheet("QGroupBox { border: 1px solid #AA00FF; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #AA00FF; }");
    QVBoxLayout *rgLayout = new QVBoxLayout(rgBox);

    QHBoxLayout *errRow = new QHBoxLayout();
    errRow->addWidget(new QLabel("Error Threshold:", this));
    rgErrorSpinner = new QDoubleSpinBox(this);
    rgErrorSpinner->setRange(1.0, 255.0);
    rgErrorSpinner->setValue(30.0);
    errRow->addWidget(rgErrorSpinner);
    rgLayout->addLayout(errRow);

    QHBoxLayout *patchRow = new QHBoxLayout();
    patchRow->addWidget(new QLabel("Patch Size:", this));
    rgPatchSizeSelect = new QComboBox(this);
    rgPatchSizeSelect->addItems({"2x2", "3x3", "4x4"});
    rgPatchSizeSelect->setCurrentText("3x3");
    patchRow->addWidget(rgPatchSizeSelect);
    rgLayout->addLayout(patchRow);

    runRegionGrowingBtn = new QPushButton("Run Region Growing", this);
    rgLayout->addWidget(runRegionGrowingBtn);
    sideLayout->addWidget(rgBox);

    // --- NEW: Dedicated Agglomerative Clustering Box ---
    QGroupBox *aggloBox = new QGroupBox("Agglomerative Clustering", this);
    aggloBox->setStyleSheet("QGroupBox { border: 1px solid #00FA9A; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #00FA9A; }");
    QVBoxLayout *aggloLayout = new QVBoxLayout(aggloBox);

    QHBoxLayout *aggloRow = new QHBoxLayout();
    aggloRow->addWidget(new QLabel("Target Clusters:", this));
    aggloClustersSpinner = new QSpinBox(this);
    aggloClustersSpinner->setRange(1, 100);
    aggloClustersSpinner->setValue(4);
    aggloRow->addWidget(aggloClustersSpinner);
    aggloLayout->addLayout(aggloRow);

    runAggloBtn = new QPushButton("Run Agglomerative", this);
    aggloLayout->addWidget(runAggloBtn);
    sideLayout->addWidget(aggloBox);
    
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
    connect(runKMeansBtn, &QPushButton::clicked, this, &MainWindow::runInteractiveKMeans);
    connect(clearSeedsBtn, &QPushButton::clicked, this, &MainWindow::clearKMeansSeeds);
    connect(runMeanShiftBtn, &QPushButton::clicked, this, &MainWindow::runMeanShift);
    connect(runRegionGrowingBtn, &QPushButton::clicked, this, &MainWindow::runRegionGrowing); // NEW
    connect(runAggloBtn, &QPushButton::clicked, this, &MainWindow::runAgglomerative);         // NEW

    log("Ready. Load an image, click on colors you want to segment, and hit a Run button!");
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

        // --- THE FIX: Calculate the centered offset margins ---
        int xOffset = (labelW - pixmapW) / 2;
        int yOffset = (labelH - pixmapH) / 2;

        // Get raw click relative to the label's top-left corner
        int rawClickX = mouseEvent->position().toPoint().x();
        int rawClickY = mouseEvent->position().toPoint().y();

        // Adjust the click to be relative to the actual drawn image pixels
        int imgClickX = rawClickX - xOffset;
        int imgClickY = rawClickY - yOffset;

        // Ensure the user actually clicked ON the image, not in the empty margins
        if (imgClickX >= 0 && imgClickX < pixmapW && imgClickY >= 0 && imgClickY < pixmapH) {
            
            // Map the scaled click back to the true original high-res image coordinates
            int trueX = (imgClickX * currentImage.cols) / pixmapW;
            int trueY = (imgClickY * currentImage.rows) / pixmapH;

            cv::Vec3b color = currentImage.at<cv::Vec3b>(trueY, trueX);
            userSeeds.push_back(color);
            userSeedPoints.push_back(cv::Point(trueX, trueY)); 

            log(QString("Picked Seed Point: (%1, %2) Color: [B:%3, G:%4, R:%5]")
                .arg(trueX).arg(trueY).arg(color[0]).arg(color[1]).arg(color[2]));
            
            // Draw the red crosshair using the adjusted image coordinates!
            drawSeedOnImage(QPoint(imgClickX, imgClickY));
            
            if (userSeeds.size() > (size_t)kSpinner->value()) {
                kSpinner->setValue(userSeeds.size());
            }
        }
        return true;
    }
    return QMainWindow::eventFilter(obj, event);
}

void MainWindow::drawSeedOnImage(QPoint pos) {
    QPixmap pm = originalImageLabel->pixmap(); 
    QPainter painter(&pm);
    painter.setPen(QPen(Qt::red, 3));
    painter.drawLine(pos.x() - 5, pos.y(), pos.x() + 5, pos.y());
    painter.drawLine(pos.x(), pos.y() - 5, pos.x(), pos.y() + 5);
    originalImageLabel->setPixmap(pm);
}

void MainWindow::clearKMeansSeeds() {
    userSeeds.clear();
    userSeedPoints.clear();
    if (!currentImage.empty()) {
        QImage qimg = cvMatToQImage(currentImage);
        originalImageLabel->setPixmap(QPixmap::fromImage(qimg).scaled(originalImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    log("Cleared all interactive seeds/points.");
}

// ---------------------------------------------------------
// Segmentation Runners
// ---------------------------------------------------------
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

void MainWindow::runMeanShift() {
    if (currentImage.empty()) { 
        log("Warning: Load an image first!"); 
        return; 
    }
    
    log("Running Mean Shift Segmentation (Please wait)...");
    QApplication::setOverrideCursor(Qt::WaitCursor);
    
    cv::Mat processedMat = currentImage.clone(); 
    double sBand = spatialBandwidthSpinner->value();
    double cBand = colorBandwidthSpinner->value();
    int iters = msIterSpinner->value();
    
    Segmentation::applyMeanShift(processedMat, sBand, cBand, iters);
    
    QApplication::restoreOverrideCursor();
    displayResult(processedMat);
    log("Finished Mean Shift Segmentation");
}

void MainWindow::runRegionGrowing() {
    if (currentImage.empty()) { 
        log("Warning: Load an image first!"); 
        return; 
    }
    if (userSeedPoints.empty()) {
        log("Warning: Please click on the original image to select at least one seed location!");
        return;
    }
    
    log(QString("Running Region Growing with %1 seeds...").arg(userSeedPoints.size()));
    QApplication::setOverrideCursor(Qt::WaitCursor);
    
    double errorThresh = rgErrorSpinner->value();
    int patchSize = 3; 
    if (rgPatchSizeSelect->currentText() == "2x2") patchSize = 2;
    else if (rgPatchSizeSelect->currentText() == "4x4") patchSize = 4;
    
    cv::Mat processedMat = Segmentation::applyRegionGrowing(currentImage, userSeedPoints, errorThresh, patchSize);
    
    QApplication::restoreOverrideCursor();
    displayResult(processedMat);
    log("Finished Region Growing Segmentation");
}

void MainWindow::runAgglomerative() {
    if (currentImage.empty()) { 
        log("Warning: Load an image first!"); 
        return; 
    }
    
    int targetK = aggloClustersSpinner->value();
    log(QString("Running Agglomerative Clustering (Target Clusters=%1). Calculating...").arg(targetK));
    
    QApplication::setOverrideCursor(Qt::WaitCursor);
    
    cv::Mat processedMat = Segmentation::applyAgglomerative(currentImage, targetK);
    
    QApplication::restoreOverrideCursor();
    displayResult(processedMat);
    log("Finished Agglomerative Clustering");
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
    cv::cvtColor(currentImage, grayImage, cv::COLOR_BGR2GRAY);
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