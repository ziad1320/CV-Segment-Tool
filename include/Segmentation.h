#pragma once
#include <opencv2/opencv.hpp>

class Segmentation {
public:
    // We pass 'k' with a default value of 4 clusters (colors)
    static cv::Mat applyKMeans(const cv::Mat& inputImage, int k = 4);
    static cv::Mat applyMeanShift(const cv::Mat& inputImage);
    static cv::Mat applyRegionGrowing(const cv::Mat& inputImage); 
    static cv::Mat applyAgglomerative(const cv::Mat& inputImage); 
};