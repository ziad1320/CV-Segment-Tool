#pragma once
#include <opencv2/opencv.hpp>
#include <QString>

class Segmentation {
public:
    static cv::Mat applyKMeans(const cv::Mat& inputImage, int k, int maxIterations, const std::vector<cv::Vec3b>& initialSeeds, QString& logOutput);
    static cv::Mat applyMeanShift(const cv::Mat& inputImage);
    static cv::Mat applyRegionGrowing(const cv::Mat& inputImage); 
    static cv::Mat applyAgglomerative(const cv::Mat& inputImage); 
};