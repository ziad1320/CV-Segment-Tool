#include "../include/Segmentation.h"
#include <QDebug>

// ==========================================
// K-MEANS ALGORITHM
// ==========================================
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

// ==========================================
// MEAN SHIFT ALGORITHM
// ==========================================
cv::Mat Segmentation::applyMeanShift(const cv::Mat& inputImage) {
    if (inputImage.empty()) return inputImage;

    cv::Mat result;
    // Mean Shift smooths out textures while preserving strict edges, creating a "cartoon" effect
    // Parameters: spatial radius (distance), color radius (color difference)
    cv::pyrMeanShiftFiltering(inputImage, result, 20.0, 40.0);
    
    qDebug() << "Mean Shift Segmentation applied.";
    return result;
}

// ==========================================
// REGION GROWING ALGORITHM
// ==========================================
cv::Mat Segmentation::applyRegionGrowing(const cv::Mat& inputImage) {
    if (inputImage.empty()) return inputImage;

    // 1. Ensure we have a 3-channel color image to draw on
    cv::Mat colorImage;
    if (inputImage.channels() == 1) {
        cv::cvtColor(inputImage, colorImage, cv::COLOR_GRAY2BGR);
    } else {
        colorImage = inputImage.clone();
    }

    // 2. Setup tracking matrices
    cv::Mat result = cv::Mat::zeros(colorImage.size(), CV_8UC3); // Blank canvas
    cv::Mat visited = cv::Mat::zeros(colorImage.size(), CV_8UC1); // Tracks where we've been

    // 3. Define the Seed (Center of the image) and the Tolerance
    cv::Point seed(colorImage.cols / 2, colorImage.rows / 2);
    cv::Vec3b seedColor = colorImage.at<cv::Vec3b>(seed);
    int tolerance = 35; // How much the color can deviate before stopping

    // 4. Setup the BFS Queue
    std::vector<cv::Point> pointQueue;
    pointQueue.push_back(seed);
    visited.at<uchar>(seed) = 255;

    // 8-Way connection directions (x, y)
    int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};

    // 5. The Growth Loop
    while (!pointQueue.empty()) {
        cv::Point current = pointQueue.front();
        pointQueue.erase(pointQueue.begin());

        // Color the grown pixel Neon Green (OpenCV uses BGR, so 0, 255, 0)
        result.at<cv::Vec3b>(current) = cv::Vec3b(0, 255, 0);

        // Check the 8 neighbors around the current pixel
        for (int i = 0; i < 8; ++i) {
            int nx = current.x + dx[i];
            int ny = current.y + dy[i];

            // Safety: Make sure neighbor is inside the image bounds
            if (nx >= 0 && nx < colorImage.cols && ny >= 0 && ny < colorImage.rows) {
                // Make sure we haven't checked this pixel already
                if (visited.at<uchar>(ny, nx) == 0) {
                    cv::Vec3b neighborColor = colorImage.at<cv::Vec3b>(ny, nx);
                    
                    // Math: Calculate absolute color difference
                    int diffB = std::abs(neighborColor[0] - seedColor[0]);
                    int diffG = std::abs(neighborColor[1] - seedColor[1]);
                    int diffR = std::abs(neighborColor[2] - seedColor[2]);
                    
                    // If the neighbor is similar to the seed, add it to the region!
                    if (diffB < tolerance && diffG < tolerance && diffR < tolerance) {
                        visited.at<uchar>(ny, nx) = 255; // Mark as visited
                        pointQueue.push_back(cv::Point(nx, ny)); // Add to queue to check ITS neighbors
                    }
                }
            }
        }
    }
    
    // 6. Blend the neon green region over the original image for a nice UI presentation
    cv::Mat finalOutput;
    cv::addWeighted(colorImage, 0.6, result, 0.4, 0, finalOutput);
    
    qDebug() << "Region Growing applied starting at seed:" << seed.x << "," << seed.y;
    return finalOutput;
}

cv::Mat Segmentation::applyAgglomerative(const cv::Mat& inputImage) {
    qDebug() << "Agglomerative Clustering not yet implemented.";
    return inputImage.clone(); 
}