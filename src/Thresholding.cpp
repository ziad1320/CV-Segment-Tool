#include "../include/Thresholding.h"
#include <QDebug>
#include <vector>
#include <algorithm>

// ==========================================
// OTSU THRESHOLDING
// ==========================================
cv::Mat Thresholding::applyOtsu(const cv::Mat& inputImage) {
    cv::Mat gray, result;
    if (inputImage.channels() == 3) cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);
    else gray = inputImage.clone();

    cv::threshold(gray, result, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return result;
}

// ==========================================
// OPTIMAL THRESHOLDING
// ==========================================
cv::Mat Thresholding::applyOptimal(const cv::Mat& inputImage) {
    cv::Mat gray, result;
    if (inputImage.channels() == 3) cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);
    else gray = inputImage.clone();

    int rows = gray.rows;
    int cols = gray.cols;

    // ==========================================
    // STEP 1: Initialization
    // Assume 4 corners are background, remainder is object
    // ==========================================
    
    // Grab the pixel values of the 4 extreme corners
    double cornerSum = gray.at<uchar>(0, 0) +               // Top-Left
                       gray.at<uchar>(0, cols - 1) +        // Top-Right
                       gray.at<uchar>(rows - 1, 0) +        // Bottom-Left
                       gray.at<uchar>(rows - 1, cols - 1);  // Bottom-Right
    
    double initialBgMean = cornerSum / 4.0;

    // Calculate the mean of the "remainder" (everything except the 4 corners)
    double totalSum = cv::sum(gray)[0];
    int totalPixels = rows * cols;
    
    double objectSum = totalSum - cornerSum;
    int objectPixels = totalPixels - 4;
    double initialFgMean = objectSum / (double)objectPixels;

    // Calculate the very first T based on these initial means
    double T = (initialBgMean + initialFgMean) / 2.0;
    double previousT = -1.0;

    // ==========================================
    // STEPS 2, 3, & 4: The Iteration Loop
    // ==========================================
    
    // Step 4: Loop until T(t+1) == T(t) 
    // We use > 0.1 for floating point safety
    while (std::abs(T - previousT) > 0.1) {
        previousT = T;

        // Step 2: Segment based on current T
        cv::Mat bgMask = gray < T;
        cv::Mat fgMask = gray >= T;

        // Step 2: Compute new means for background and object
        cv::Scalar bgMean = cv::mean(gray, bgMask);
        cv::Scalar fgMean = cv::mean(gray, fgMask);

        // Step 3: Set new T
        T = (bgMean[0] + fgMean[0]) / 2.0;
    }

    qDebug() << "Optimal Threshold Calculated:" << T;

    // Apply the final threshold
    cv::threshold(gray, result, T, 255, cv::THRESH_BINARY);
    return result;
}

// ==========================================
// LOCAL THRESHOLDING
// ==========================================
cv::Mat Thresholding::applyLocal(const cv::Mat& inputImage) {
    cv::Mat gray, result;
    if (inputImage.channels() == 3) cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);
    else gray = inputImage.clone();

    // Adaptive Thresholding: 
    // - Uses a 15x15 pixel local window
    // - Subtracts a constant 'C' of 5 to fine-tune noise
    cv::adaptiveThreshold(gray, result, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 15, 5);
    
    return result;
}

// ==========================================
// SPECTRAL THRESHOLDING
// ==========================================
cv::Mat Thresholding::applySpectral(const cv::Mat& inputImage) {
    cv::Mat gray;
    if (inputImage.channels() == 3) cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);
    else gray = inputImage.clone();

    // 1. Flatten the image into a 1D column of pixels for clustering
    cv::Mat data = gray.reshape(1, gray.total());
    data.convertTo(data, CV_32F);

    // 2. Run K-Means to find 3 distinct brightness modes
    int K = 3; 
    cv::Mat labels, centers;
    cv::kmeans(data, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

    // 3. Sort the centers to figure out which cluster is Dark, Mid, and Bright
    std::vector<std::pair<float, int>> sortedCenters;
    for (int i = 0; i < K; ++i) {
        sortedCenters.push_back({centers.at<float>(i), i});
    }
    std::sort(sortedCenters.begin(), sortedCenters.end());

    // 4. Assign visual colors: Black (0), Gray (127), White (255)
    std::vector<uchar> colors = {0, 127, 255}; 
    std::vector<int> labelToColor(K);
    for (int i = 0; i < K; ++i) {
        labelToColor[sortedCenters[i].second] = colors[i];
    }

    // 5. Reconstruct the image with the new 3-mode colors
    cv::Mat result(gray.size(), CV_8UC1);
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            int idx = i * result.cols + j;
            int cluster_idx = labels.at<int>(idx);
            result.at<uchar>(i, j) = labelToColor[cluster_idx];
        }
    }
    return result;
}