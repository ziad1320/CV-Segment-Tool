#include "../include/Segmentation.h"
#include <QDebug>
#include <queue>
#include <vector>
#include <limits>
#include <cmath>

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
const double CONVERGENCE_THRESHOLD = 0.5; // for shifting feature vectors
const double MERGE_THRESHOLD = 5.0; // for merging close clusters in L*a*b* space

void Segmentation::applyMeanShift(cv::Mat& image, double spatialBandwidth, double colorBandwidth, int maxIterations) {
    // 1. Prepare data and convert to L*a*b* color space
    cv::Mat labImage;
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);
    labImage.convertTo(labImage, CV_64FC3);

    int rows = labImage.rows;
    int cols = labImage.cols;
    int numPoints = rows * cols;

    // 2. Pre-scale feature space to simplify distance checks
    cv::Mat scaledSpace(numPoints, 5, CV_64F);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            cv::Vec3d lab = labImage.at<cv::Vec3d>(r, c);
            scaledSpace.at<double>(r * cols + c, 0) = lab[0] / colorBandwidth;
            scaledSpace.at<double>(r * cols + c, 1) = lab[1] / colorBandwidth;
            scaledSpace.at<double>(r * cols + c, 2) = lab[2] / colorBandwidth;
            scaledSpace.at<double>(r * cols + c, 3) = static_cast<double>(c) / spatialBandwidth;
            scaledSpace.at<double>(r * cols + c, 4) = static_cast<double>(r) / spatialBandwidth;
        }
    }

    // 3. Optimization: Track visited points to drastically reduce iterations
    std::vector<bool> visited(numPoints, false);
    std::vector<cv::Mat> clusterModes;
    std::vector<int> pointClusterMap(numPoints, -1);
    std::vector<int> trackedPoints;
    trackedPoints.reserve(numPoints); 

    // 4. Mean Shift execution matching your algorithm image
    for (int i = 0; i < numPoints; ++i) {
        if (visited[i]) continue; // Skip points we've already grouped

        cv::Mat currentPoint = scaledSpace.row(i).clone();
        
        for (int iter = 0; iter < maxIterations; ++iter) {
            cv::Mat num = cv::Mat::zeros(1, 5, CV_64F);
            double den = 0.0;
            trackedPoints.clear();

            // Establish search window bounds
            int pt_c = static_cast<int>(currentPoint.at<double>(0, 3) * spatialBandwidth);
            int pt_r = static_cast<int>(currentPoint.at<double>(0, 4) * spatialBandwidth);
            
            int c_start = std::max(0, pt_c - static_cast<int>(spatialBandwidth));
            int c_end = std::min(cols - 1, pt_c + static_cast<int>(spatialBandwidth));
            int r_start = std::max(0, pt_r - static_cast<int>(spatialBandwidth));
            int r_end = std::min(rows - 1, pt_r + static_cast<int>(spatialBandwidth));

            // Find neighbors within bandwidth
            for (int r = r_start; r <= r_end; ++r) {
                for (int c = c_start; c <= c_end; ++c) {
                    int idx = r * cols + c;
                    cv::Mat neighbor = scaledSpace.row(idx);
                    
                    double spatialDistSq = std::pow(neighbor.at<double>(0, 3) - currentPoint.at<double>(0, 3), 2) + 
                                           std::pow(neighbor.at<double>(0, 4) - currentPoint.at<double>(0, 4), 2);
                    double colorDistSq = std::pow(neighbor.at<double>(0, 0) - currentPoint.at<double>(0, 0), 2) + 
                                         std::pow(neighbor.at<double>(0, 1) - currentPoint.at<double>(0, 1), 2) + 
                                         std::pow(neighbor.at<double>(0, 2) - currentPoint.at<double>(0, 2), 2);

                    if (spatialDistSq <= 1.0 && colorDistSq <= 1.0) {
                        num += neighbor;
                        den += 1.0;
                        trackedPoints.push_back(idx); // Track all points falling in the window
                    }
                }
            }

            if (den == 0.0) break;
            
            cv::Mat nextPoint = num / den;
            double shiftDist = cv::norm(nextPoint - currentPoint);
            currentPoint = nextPoint;

            if (shiftDist < 0.5) break; // Converged
            
        } // End of single path iteration

        // 5. Check Merge Condition from your algorithm
        bool merged = false;
        for (size_t cIdx = 0; cIdx < clusterModes.size(); ++cIdx) {
            // Check if distance < 0.5 * Bandwidth (Since space is scaled, bandwidth is effectively 1)
            if (cv::norm(clusterModes[cIdx] - currentPoint) < 0.5) { 
                clusterModes[cIdx] = (clusterModes[cIdx] + currentPoint) / 2.0; // Mean of cluster c
                
                // Cluster all tracked points to cluster c
                for (int tp : trackedPoints) {
                    if (!visited[tp]) {
                        visited[tp] = true;
                        pointClusterMap[tp] = static_cast<int>(cIdx);
                    }
                }
                merged = true;
                break;
            }
        }

        // 6. No Merge: Create new cluster
        if (!merged) {
            clusterModes.push_back(currentPoint);
            int newIdx = static_cast<int>(clusterModes.size()) - 1;
            
            // Cluster all tracked points to new mean
            for (int tp : trackedPoints) {
                if (!visited[tp]) {
                    visited[tp] = true;
                    pointClusterMap[tp] = newIdx;
                }
            }
        }
    }

    // 7. Paint the final image using the real cluster colors
    for (int i = 0; i < numPoints; ++i) {
        int clusterIdx = pointClusterMap[i];
        if (clusterIdx == -1) clusterIdx = 0; // Failsafe

        cv::Mat mode = clusterModes[clusterIdx];
        
        // De-scale the color values
        double L = std::max(0.0, std::min(255.0, mode.at<double>(0, 0) * colorBandwidth));
        double a = std::max(0.0, std::min(255.0, mode.at<double>(0, 1) * colorBandwidth));
        double b = std::max(0.0, std::min(255.0, mode.at<double>(0, 2) * colorBandwidth));

        // Convert the L*a*b* cluster mode back to BGR to paint the pixel
        cv::Mat labMat(1, 1, CV_8UC3, cv::Scalar(L, a, b));
        cv::Mat bgrMat;
        cv::cvtColor(labMat, bgrMat, cv::COLOR_Lab2BGR);
        
        image.at<cv::Vec3b>(i / cols, i % cols) = bgrMat.at<cv::Vec3b>(0, 0);
    }
}

// ==========================================
// REGION GROWING ALGORITHM
// ==========================================
#include <queue>

cv::Mat Segmentation::applyRegionGrowing(const cv::Mat& inputImage, const std::vector<cv::Point>& seeds, double errorThreshold, int patchSize) {
    if (seeds.empty()) return inputImage.clone();

    // output image starts completely black
    cv::Mat output = cv::Mat::zeros(inputImage.size(), inputImage.type());
    // Visited mask tracks allocated and rejected pixels
    cv::Mat visited = cv::Mat::zeros(inputImage.size(), CV_8UC1);
    
    // Calculate neighbor offset based on patch size (e.g., 3x3 means offset of 1)
    int offset = patchSize / 2; 

    struct Region {
        cv::Vec3d meanColor;
        int count;
        cv::Vec3b displayColor;
    };
    
    std::vector<Region> regions;
    std::vector<std::queue<cv::Point>> regionQueues;
    
    // Initialize regions with seeds
    for (size_t i = 0; i < seeds.size(); ++i) {
        cv::Point pt = seeds[i];
        if (pt.x < 0 || pt.x >= inputImage.cols || pt.y < 0 || pt.y >= inputImage.rows) continue;
        
        Region r;
        r.meanColor = inputImage.at<cv::Vec3b>(pt); // each region mean = seed value
        r.count = 1;
        r.displayColor = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256); // Random distinct color
        regions.push_back(r);
        
        std::queue<cv::Point> q;
        q.push(pt);
        regionQueues.push_back(q);
        
        visited.at<uchar>(pt) = 1;
        output.at<cv::Vec3b>(pt) = r.displayColor;
    }
    
    bool pixelsAllocated = true;
    
    // while there is unallocated pixels (that can be reached)
    while (pixelsAllocated) {
        pixelsAllocated = false;
        
        for (size_t i = 0; i < regionQueues.size(); ++i) {
            std::queue<cv::Point>& q = regionQueues[i];
            int size = q.size(); 
            
            // for each pixel in each region
            for (int k = 0; k < size; ++k) {
                cv::Point p = q.front();
                q.pop();
                
                // if unallocated neighbors of patches
                for (int dy = -offset; dy <= offset; ++dy) {
                    for (int dx = -offset; dx <= offset; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        
                        int nx = p.x + dx;
                        int ny = p.y + dy;
                        
                        if (nx >= 0 && nx < inputImage.cols && ny >= 0 && ny < inputImage.rows) {
                            if (visited.at<uchar>(ny, nx) == 0) { // Check if unallocated
                                cv::Vec3b neighborColor = inputImage.at<cv::Vec3b>(ny, nx);
                                
                                // set error as similarity measure (Euclidean distance in color space)
                                double dist = cv::norm(regions[i].meanColor - cv::Vec3d(neighborColor[0], neighborColor[1], neighborColor[2]));
                                
                                if (dist <= errorThreshold) {
                                    // then add to region
                                    visited.at<uchar>(ny, nx) = 1;
                                    output.at<cv::Vec3b>(ny, nx) = regions[i].displayColor;
                                    q.push(cv::Point(nx, ny));
                                    
                                    // and recalculate region mean
                                    regions[i].meanColor = (regions[i].meanColor * regions[i].count + cv::Vec3d(neighborColor[0], neighborColor[1], neighborColor[2])) / (regions[i].count + 1.0);
                                    regions[i].count++;
                                    
                                    pixelsAllocated = true;
                                } else {
                                    // else mark as visited and dont add to region
                                    visited.at<uchar>(ny, nx) = 1; 
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return output;
}


// ==========================================
// AGGLOMERATIVE ALGORITHM
// ==========================================
cv::Mat Segmentation::applyAgglomerative(const cv::Mat& inputImage, int targetClusters) {
    if (inputImage.empty() || targetClusters <= 0) return inputImage.clone();

    // 1. The Built-in Trick: Use K-Means to squash 160,000 pixels into 50 color "buckets"
    int initialK = std::min(50, inputImage.rows * inputImage.cols); 
    if (targetClusters >= initialK) return inputImage.clone();

    // Prepare data for OpenCV's built-in kmeans
    cv::Mat data = inputImage.reshape(1, inputImage.cols * inputImage.rows);
    data.convertTo(data, CV_32F);

    cv::Mat labels, centers;
    cv::kmeans(data, initialK, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);

    // 2. Setup Your Agglomerative Logic on the 50 buckets
    struct Cluster {
        cv::Vec3d colorSum;
        int count;
        std::vector<int> pixelIndices; // Store exact pixel locations
    };
    
    std::vector<Cluster> clusters(initialK);
    std::vector<cv::Vec3d> means(initialK);

    // Sort all the pixels into their initial 50 clusters
    for (int i = 0; i < labels.rows; ++i) {
        int clusterIdx = labels.at<int>(i);
        cv::Vec3b col = inputImage.at<cv::Vec3b>(i / inputImage.cols, i % inputImage.cols);
        
        clusters[clusterIdx].colorSum += cv::Vec3d(col[0], col[1], col[2]);
        clusters[clusterIdx].count++;
        clusters[clusterIdx].pixelIndices.push_back(i);
    }

    // Calculate the mean color of the 50 buckets
    for (int i = 0; i < initialK; ++i) {
        if (clusters[i].count > 0) {
            means[i] = clusters[i].colorSum / static_cast<double>(clusters[i].count);
        }
    }

    int currentClusters = initialK;
    
    // 3. Run YOUR exact Agglomerative steps (this takes milliseconds now!)
    while (currentClusters > targetClusters) {
        double minDistSq = std::numeric_limits<double>::max();
        int bestA = -1, bestB = -1;

        // Find the closest pair of clusters
        for (int i = 0; i < initialK; ++i) {
            if (clusters[i].count == 0) continue; 
            
            for (int j = i + 1; j < initialK; ++j) {
                if (clusters[j].count == 0) continue;

                double distSq = std::pow(means[i][0] - means[j][0], 2) + 
                                std::pow(means[i][1] - means[j][1], 2) + 
                                std::pow(means[i][2] - means[j][2], 2);

                if (distSq < minDistSq) {
                    minDistSq = distSq;
                    bestA = i;
                    bestB = j;
                }
            }
        }

        // Merge Cluster B into Cluster A
        clusters[bestA].colorSum += clusters[bestB].colorSum;
        clusters[bestA].count += clusters[bestB].count;
        
        // Transfer all pixel locations from B to A
        clusters[bestA].pixelIndices.insert(clusters[bestA].pixelIndices.end(), 
                                            clusters[bestB].pixelIndices.begin(), 
                                            clusters[bestB].pixelIndices.end());

        // Update the new mean
        means[bestA] = clusters[bestA].colorSum / static_cast<double>(clusters[bestA].count);
        
        // Deactivate Cluster B
        clusters[bestB].count = 0;
        clusters[bestB].pixelIndices.clear();

        currentClusters--;
    }

    // 4. Paint the final high-resolution output image
    cv::Mat finalOutput = inputImage.clone();
    for (int i = 0; i < initialK; ++i) {
        if (clusters[i].count > 0) {
            cv::Vec3b meanColor(static_cast<uchar>(means[i][0]), 
                                static_cast<uchar>(means[i][1]), 
                                static_cast<uchar>(means[i][2]));
            
            // Paint every pixel that belongs to this merged cluster
            for (int idx : clusters[i].pixelIndices) {
                finalOutput.at<cv::Vec3b>(idx / inputImage.cols, idx % inputImage.cols) = meanColor;
            }
        }
    }

    return finalOutput;
}