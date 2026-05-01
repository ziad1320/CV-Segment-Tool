#include "../include/Segmentation.h"
#include <QDebug>

cv::Mat Segmentation::applyKMeans(const cv::Mat& inputImage, int k) {
    if (inputImage.empty()) return inputImage;

    // 1. Flatten the image into a 1D column of pixels (3 channels for RGB)
    cv::Mat data = inputImage.reshape(1, inputImage.total());
    data.convertTo(data, CV_32F); // K-Means requires 32-bit floats

    // 2. Define K-Means criteria and run the algorithm
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0);
    cv::Mat labels, centers;
    cv::kmeans(data, k, labels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

    // 3. Convert the centers (the mathematical average colors) back to standard 8-bit colors
    centers.convertTo(centers, CV_8U);

    // 4. Create a new image by mapping every pixel to its new cluster color
    cv::Mat result(inputImage.size(), inputImage.type());
    cv::Mat resultFlattened = result.reshape(1, result.total());

    for (int i = 0; i < data.rows; ++i) {
        int clusterIdx = labels.at<int>(i);
        // Apply the BGR color of the assigned cluster to the pixel
        resultFlattened.at<cv::Vec3b>(i, 0)[0] = centers.at<uchar>(clusterIdx, 0); // B
        resultFlattened.at<cv::Vec3b>(i, 0)[1] = centers.at<uchar>(clusterIdx, 1); // G
        resultFlattened.at<cv::Vec3b>(i, 0)[2] = centers.at<uchar>(clusterIdx, 2); // R
    }

    qDebug() << "K-Means Segmentation applied with K =" << k;
    return result;
}

cv::Mat Segmentation::applyMeanShift(const cv::Mat& inputImage) {
    if (inputImage.empty()) return inputImage;

    cv::Mat result;
    // Mean Shift smooths out textures while preserving strict edges, creating a "cartoon" effect
    // Parameters: spatial radius (distance), color radius (color difference)
    cv::pyrMeanShiftFiltering(inputImage, result, 20.0, 40.0);
    
    qDebug() << "Mean Shift Segmentation applied.";
    return result;
}

cv::Mat Segmentation::applyRegionGrowing(const cv::Mat& inputImage) {
    qDebug() << "Region Growing not yet implemented.";
    return inputImage.clone(); // Returns original image for now
}

cv::Mat Segmentation::applyAgglomerative(const cv::Mat& inputImage) {
    qDebug() << "Agglomerative Clustering not yet implemented.";
    return inputImage.clone(); // Returns original image for now
}