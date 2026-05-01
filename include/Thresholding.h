#pragma once
#include <opencv2/opencv.hpp>

class Thresholding {
public:
    static cv::Mat applyOtsu(const cv::Mat& inputImage);
    static cv::Mat applyOptimal(const cv::Mat& inputImage);
    static cv::Mat applySpectral(const cv::Mat& inputImage); // Multi-mode (3 levels)
    static cv::Mat applyLocal(const cv::Mat& inputImage);    // Adaptive
};