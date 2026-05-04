#pragma once
#include <opencv2/opencv.hpp>
#include <QString>

class Thresholding {
public:
    static cv::Mat applyOtsu(const cv::Mat& grayImage, QString& logOutput);
    static cv::Mat applyOptimal(const cv::Mat& grayImage, QString& logOutput);
    static cv::Mat applySpectral(const cv::Mat& grayImage); 
    static cv::Mat applyLocal(const cv::Mat& grayImage, int blockSize, double C);
};