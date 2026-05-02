#include "../include/Segmentation.h"
#include <QDebug>

// ==========================================
// K-MEANS ALGORITHM
// ==========================================
cv::Mat Segmentation::applyKMeans(const cv::Mat& inputImage, int k, int maxIterations, const std::vector<cv::Vec3b>& initialSeeds, QString& logOutput) {
    if (inputImage.empty()) return inputImage;

    int rows = inputImage.rows;
    int cols = inputImage.cols;
    cv::Mat result = inputImage.clone();

    // 1. Initialize Centroids (Centers)
    std::vector<cv::Vec3f> centers(k);
    
    // If user clicked points, use those colors! Otherwise, pick random pixels.
    for (int i = 0; i < k; ++i) {
        if (i < initialSeeds.size()) {
            centers[i] = cv::Vec3f(initialSeeds[i][0], initialSeeds[i][1], initialSeeds[i][2]);
        } else {
            // Pick a random pixel if the user didn't click enough points
            int r = rand() % rows;
            int c = rand() % cols;
            cv::Vec3b pixel = inputImage.at<cv::Vec3b>(r, c);
            centers[i] = cv::Vec3f(pixel[0], pixel[1], pixel[2]);
        }
    }

    // Array to hold the assigned cluster (0 to k-1) for every single pixel
    std::vector<int> labels(rows * cols, 0);
    bool centersChanged = true;
    int currentIter = 0;

    // 2. The K-Means Loop
    while (centersChanged && currentIter < maxIterations) {
        centersChanged = false;
        
        // Track the sums and counts to calculate the new average later
        std::vector<cv::Vec3f> newCenterSums(k, cv::Vec3f(0, 0, 0));
        std::vector<int> clusterCounts(k, 0);

        // Step A: Assign every pixel to the nearest center
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                cv::Vec3b pixel = inputImage.at<cv::Vec3b>(r, c);
                
                float minDistance = FLT_MAX;
                int bestCluster = 0;

                // Test distance against all 'k' centers
                for (int i = 0; i < k; ++i) {
                    float dB = pixel[0] - centers[i][0];
                    float dG = pixel[1] - centers[i][1];
                    float dR = pixel[2] - centers[i][2];
                    
                    // Euclidean distance (we don't need sqrt since we just need the smallest)
                    float dist = (dB*dB) + (dG*dG) + (dR*dR); 

                    if (dist < minDistance) {
                        minDistance = dist;
                        bestCluster = i;
                    }
                }

                // Save assignment and add to running totals
                int pixelIdx = r * cols + c;
                labels[pixelIdx] = bestCluster;
                newCenterSums[bestCluster] += cv::Vec3f(pixel[0], pixel[1], pixel[2]);
                clusterCounts[bestCluster]++;
            }
        }

        // Step B: Recalculate the centers
        for (int i = 0; i < k; ++i) {
            if (clusterCounts[i] > 0) {
                cv::Vec3f updatedCenter = newCenterSums[i] / (float)clusterCounts[i];
                
                // Check if the center moved significantly (> 1.0 color value)
                float moveDist = cv::norm(centers[i] - updatedCenter);
                if (moveDist > 1.0) {
                    centersChanged = true;
                }
                centers[i] = updatedCenter;
            }
        }
        currentIter++;
    }

    // 3. Rebuild the final image using the final average colors
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int bestCluster = labels[r * cols + c];
            result.at<cv::Vec3b>(r, c) = cv::Vec3b(
                (uchar)centers[bestCluster][0], 
                (uchar)centers[bestCluster][1], 
                (uchar)centers[bestCluster][2]
            );
        }
    }

    logOutput = QString("K-Means converged in %1 iterations with K=%2").arg(currentIter).arg(k);
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


// ==========================================
// AGGLOMERATIVE ALGORITHM
// ==========================================
cv::Mat Segmentation::applyAgglomerative(const cv::Mat& inputImage) {
    if (inputImage.empty()) return inputImage;

    // 1. Downsample heavily. 
    // Doing this on a full image requires a supercomputer. We downscale to 20x20.
    int grid = 20;
    cv::Mat smallImg;
    cv::resize(inputImage, smallImg, cv::Size(grid, grid));

    // 2. Setup initial clusters (every pixel is its own cluster to start)
    int N = grid * grid;
    std::vector<int> labels(N);
    std::vector<cv::Vec3f> centers(N);
    std::vector<int> counts(N, 1);

    for (int i = 0; i < N; ++i) {
        labels[i] = i; // Everyone is their own boss initially
        int r = i / grid;
        int c = i % grid;
        cv::Vec3b pix = smallImg.at<cv::Vec3b>(r, c);
        centers[i] = cv::Vec3f(pix[0], pix[1], pix[2]);
    }

    // 3. Loop and Merge until we reach K dominant clusters
    int K = 4; // We want 4 final colors
    int currentClusters = N;

    while (currentClusters > K) {
        float minDistance = FLT_MAX;
        int bestA = -1;
        int bestB = -1;

        // Find the two closest clusters in color space
        for (int i = 0; i < N; ++i) {
            if (labels[i] != i) continue; // Only look at active cluster heads
            for (int j = i + 1; j < N; ++j) {
                if (labels[j] != j) continue;
                
                // Calculate Euclidean distance between colors
                float dB = centers[i][0] - centers[j][0];
                float dG = centers[i][1] - centers[j][1];
                float dR = centers[i][2] - centers[j][2];
                float dist = (dB*dB + dG*dG + dR*dR);

                if (dist < minDistance) {
                    minDistance = dist;
                    bestA = i;
                    bestB = j;
                }
            }
        }

        // Merge Cluster B into Cluster A
        labels[bestB] = bestA;
        currentClusters--;

        // Update all blocks that used to point to B, to now point to A
        for (int i = 0; i < N; ++i) {
            if (labels[i] == bestB) {
                labels[i] = bestA;
            }
        }

        // Recalculate the new average color of the merged cluster A
        float newB = (centers[bestA][0] * counts[bestA] + centers[bestB][0] * counts[bestB]) / (counts[bestA] + counts[bestB]);
        float newG = (centers[bestA][1] * counts[bestA] + centers[bestB][1] * counts[bestB]) / (counts[bestA] + counts[bestB]);
        float newR = (centers[bestA][2] * counts[bestA] + centers[bestB][2] * counts[bestB]) / (counts[bestA] + counts[bestB]);
        
        centers[bestA] = cv::Vec3f(newB, newG, newR);
        counts[bestA] += counts[bestB];
    }

    // 4. Paint the small 20x20 grid with the final 4 clustered colors
    cv::Mat clusteredSmall = cv::Mat::zeros(grid, grid, CV_8UC3);
    for (int i = 0; i < N; ++i) {
        int head = labels[i];
        int r = i / grid;
        int c = i % grid;
        clusteredSmall.at<cv::Vec3b>(r, c) = cv::Vec3b((uchar)centers[head][0], (uchar)centers[head][1], (uchar)centers[head][2]);
    }

    // 5. Upscale back to original size. 
    // We use INTER_NEAREST so it doesn't blur the edges of our mathematical clusters.
    cv::Mat result;
    cv::resize(clusteredSmall, result, inputImage.size(), 0, 0, cv::INTER_NEAREST);

    qDebug() << "Agglomerative Clustering applied. Reduced" << N << "starting points down to" << currentClusters << "clusters.";
    return result;
}