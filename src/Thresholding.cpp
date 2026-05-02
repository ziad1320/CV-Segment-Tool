#include "../include/Thresholding.h"
#include <QDebug>
#include <vector>
#include <algorithm>

// ==========================================
// OTSU THRESHOLDING
// ==========================================
/*The goal of Otsu's method is to find the perfect threshold value (k) to separate the pixels of an image into two 
    distinct classes (foreground and background). It does this by testing every possible threshold and picking the 
    one that maximizes the between-class variance, Maximizing this variance simply means we want the foreground and 
    background to be as mathematically distinct from each other as possible.*/
cv::Mat Thresholding::applyOtsu(const cv::Mat& inputImage, QString& logOutput) {
    cv::Mat gray, result;
    if (inputImage.channels() == 3) cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);
    else gray = inputImage.clone();

    int totalPixels = gray.rows * gray.cols;

    // ==========================================
    // STEP 1: Compute normalized histogram (Pi)
    // ==========================================
    std::vector<double> P(256, 0.0);
    for (int r = 0; r < gray.rows; ++r) { 
        for (int c = 0; c < gray.cols; ++c) {
            int pixelValue = gray.at<uchar>(r, c);
            P[pixelValue] += 1.0;
        }
    }
    // Divide by total pixels to get probabilities (normalisation)
    for (int i = 0; i < 256; ++i) {
        P[i] = P[i] / totalPixels; 
    }

    // ==========================================
    // STEP 4: Compute global mean (mG)
    // We calculate this early because it is a constant
    // ==========================================
    double mG = 0.0;
    for (int i = 0; i < 256; ++i) {
        mG += i * P[i];
    }

    // Variables to track our search for the best threshold
    double maxVariance = -1.0;
    int optimalThreshold = 0;

    // Running totals for our cumulative math
    double P1_k = 0.0;  //background probability
    double m_k = 0.0;  //cumulative mean

    // ==========================================
    // STEPS 2, 3, 5, & 6: Test all thresholds (k)
    // ==========================================
    // We start a loop to test every single number from 0 to 255 to see which one makes the best threshold line (k)
    for (int k = 0; k < 256; ++k) {
        // Step 2: Compute cumulative probability P1(k)
        P1_k += P[k]; 
        // Step 3: Compute cumulative mean m(k)
        m_k += k * P[k];
        // Safety check to avoid dividing by zero
        if (P1_k == 0.0 || P1_k == 1.0) {
            continue; 
        }
        // Step 5: Compute between-class variance (sigma_b_squared)
        // Formula: [mG * P1(k) - m(k)]^2 / [P1(k) * (1 - P1(k))]
        double numerator = (mG * P1_k) - m_k;
        numerator = numerator * numerator; // Square it
        
        double denominator = P1_k * (1.0 - P1_k);
        
        double sigma_b_squared = numerator / denominator;

        // Step 6: Find the threshold 'k' that maximizes the variance
        if (sigma_b_squared > maxVariance) {
            maxVariance = sigma_b_squared;
            optimalThreshold = k;
        }
    }
    logOutput = QString("Otsu Threshold Calculated: %1").arg(optimalThreshold);

    // Now apply our manually calculated optimal threshold!
    cv::threshold(gray, result, optimalThreshold, 255, cv::THRESH_BINARY);
    return result;
}

// ==========================================
// OPTIMAL THRESHOLDING
// ==========================================
cv::Mat Thresholding::applyOptimal(const cv::Mat& inputImage, QString& logOutput) {
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
    logOutput = QString("Optimal Threshold Calculated: %1").arg(T);
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