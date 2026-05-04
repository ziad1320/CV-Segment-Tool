#pragma once

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QTextEdit>
#include <QGroupBox>
#include <QSpinBox>
#include <opencv2/opencv.hpp>
#include <vector> // NEW: For storing our clicked points

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow() = default;

protected:
    // NEW: This intercepts mouse clicks on our image
    bool eventFilter(QObject *obj, QEvent *event) override;

private slots:
    void openImage();
    void processThreshold();   
    void runLocalThreshold();

    void runInteractiveKMeans(); 
    void clearKMeansSeeds();     
    void runMeanShift();
    void runRegionGrowing();
    void runAgglomerative();

private:
    // UI Elements - Images
    QLabel *originalImageLabel;
    QLabel *processedImageLabel;

    // UI Elements - General
    QPushButton *loadButton;
    QTextEdit *logTerminal;

    // UI Elements - Thresholding
    QComboBox *thresholdSelect;
    QPushButton *applyThresholdBtn;

    // UI Elements - Local Threshold Box
    QSpinBox *localBlockSpinner;
    QDoubleSpinBox *localCSpinner;
    QPushButton *runLocalThresholdBtn;

    // UI Elements - Segmentation
    QDoubleSpinBox *spatialBandwidthSpinner;
    QDoubleSpinBox *colorBandwidthSpinner;
    QSpinBox *msIterSpinner;
    QPushButton *runMeanShiftBtn;

    // UI Elements - Interactive K-Means Box
    QSpinBox *kSpinner;
    QSpinBox *iterSpinner;
    QPushButton *runKMeansBtn;
    QPushButton *clearSeedsBtn;

    // Region Growing Elements
    QDoubleSpinBox *rgErrorSpinner;
    QComboBox *rgPatchSizeSelect;
    QPushButton *runRegionGrowingBtn;

    // Agglomerative Elements
    QSpinBox *aggloClustersSpinner;
    QPushButton *runAggloBtn;

    // Data
    cv::Mat currentImage;
    std::vector<cv::Vec3b> userSeeds;
    std::vector<cv::Point> userSeedPoints;

    // Helper functions
    void log(const QString &message);
    QImage cvMatToQImage(const cv::Mat &inMat);
    void displayResult(const cv::Mat &img);
    void drawSeedOnImage(QPoint pos); // Draws a visual marker where clicked
};