#pragma once
#include <opencv2/opencv.hpp>
#include <QString>

class Segmentation {
public:
    static cv::Mat applyKMeans(const cv::Mat& inputImage, int k, int maxIterations, const std::vector<cv::Vec3b>& initialSeeds, QString& logOutput);
    static void applyMeanShift(cv::Mat& inputImage, double spatialBandwidth, double colorBandwidth, int maxIterations);
    static cv::Mat applyRegionGrowing(const cv::Mat& inputImage, const std::vector<cv::Point>& seeds, double errorThreshold, int patchSize);
    static cv::Mat applyAgglomerative(const cv::Mat& inputImage, int targetClusters);
};