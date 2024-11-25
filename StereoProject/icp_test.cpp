#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// Euclidean distance between two cv::Point2f
double euclideanDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2)); // d_e = sqrt(d_x^2 + d_y^2)
}

// Nearest neighbor correspondences
std::vector<int> findCorrespondences(const std::vector<cv::Point2f>& source, const std::vector<cv::Point2f>& target) {
    std::vector<int> correspondences(source.size(), -1);

    for (size_t i = 0; i < source.size(); ++i) {
        double minDist = std::numeric_limits<double>::max();
        int bestMatch = -1;

        for (size_t j = 0; j < target.size(); ++j) {
            double dist = euclideanDistance(source[i], target[j]);
            if (dist < minDist) {
                minDist = dist;
                bestMatch = j;
            }
        }
        correspondences[i] = bestMatch;
    std::cout << bestMatch << std::endl;
    }
    return correspondences;
}

// Function to compute the rigid transformation (rotation + translation)
cv::Mat computeTransformation(const std::vector<cv::Point2f>& source, const std::vector<cv::Point2f>& target, const std::vector<int>& correspondences) {
    // Compute centroids
    cv::Point2f sourceCentroid(0, 0), targetCentroid(0, 0);
    int validCorrespondences = 0;

    for (size_t i = 0; i < source.size(); ++i) {
        int matchIdx = correspondences[i];
        if (matchIdx >= 0) {
            sourceCentroid += source[i];
            targetCentroid += target[matchIdx];
            validCorrespondences++;
        }
    }

    sourceCentroid *= (1.0 / validCorrespondences);
    targetCentroid *= (1.0 / validCorrespondences);

    // Compute cross-covariance matrix
    cv::Mat H = cv::Mat::zeros(2, 2, CV_64F);
    for (size_t i = 0; i < source.size(); ++i) {
        int matchIdx = correspondences[i];
        if (matchIdx >= 0) {
            cv::Point2f srcPoint = source[i] - sourceCentroid;
            cv::Point2f tgtPoint = target[matchIdx] - targetCentroid;

            H.at<double>(0, 0) += srcPoint.x * tgtPoint.x;
            H.at<double>(0, 1) += srcPoint.x * tgtPoint.y;
            H.at<double>(1, 0) += srcPoint.y * tgtPoint.x;
            H.at<double>(1, 1) += srcPoint.y * tgtPoint.y;
        }
    }

    // Compute SVD of H
    cv::Mat U, S, Vt;
    cv::SVD::compute(H, S, U, Vt);

    // Compute rotation
    cv::Mat R = Vt.t() * U.t();

    if (cv::determinant(R) < 0) {
        std::cout << "reflection!" << std::endl;
        Vt.row(1) *= -1; // Fix reflection by flipping the sign of the last row of Vt
        R = Vt.t() * U.t();
    }

    // Compute translation
    cv::Mat t = (cv::Mat_<double>(2, 1) << targetCentroid.x, targetCentroid.y) -
        R * (cv::Mat_<double>(2, 1) << sourceCentroid.x, sourceCentroid.y);

    // Combine rotation and translation into a transformation matrix
    cv::Mat transformation = cv::Mat::eye(3, 3, CV_64F);
    R.copyTo(transformation(cv::Rect(0, 0, 2, 2)));
    transformation.at<double>(0, 2) = t.at<double>(0, 0);
    transformation.at<double>(1, 2) = t.at<double>(1, 0);

    return transformation;
}

void decomposeTransformation(const cv::Mat& transformation) {
    // Extract translation vector
    double tx = transformation.at<double>(0, 2);
    double ty = transformation.at<double>(1, 2);
    std::cout << "Translation Vector:\n";
    std::cout << "[ " << tx << ", " << ty << " ]" << std::endl;
    // Extract rotation matrix elements
    double r11 = transformation.at<double>(0, 0);
    double r21 = transformation.at<double>(1, 0);
    // Compute the rotation angle (in radians)
    double rotationAngle = std::atan2(r21, r11);
    std::cout << "Rotation Angle (in degrees): " << rotationAngle * (180.0 / CV_PI) << std::endl;
}


int main3() {
    // Example source and target point clouds
    std::vector<cv::Point2f> sourcePoints = { {2530, 690},{1468, 1673},{3412, 1636},{2487, 2466} };
    std::vector<cv::Point2f> targetPoints = { {1833.1335, 745.4787},{807.12097, 1765.9076},{2748.5408, 1658.0874},{1854.14, 2521.8276} };

    // Iterative Closest Point Algorithm
    cv::Mat transformation = cv::Mat::eye(3, 3, CV_64F); // Initial transformation

    for (int iter = 0; iter < 10; ++iter) {
        // Find nearest correspondences
        std::vector<int> correspondences = findCorrespondences(sourcePoints, targetPoints);

        // Compute transformation
        cv::Mat deltaTransform = computeTransformation(sourcePoints, targetPoints, correspondences);
        std::cout << "Delta Transform:\n" << deltaTransform << std::endl;

        // Update total transformation
        transformation = deltaTransform * transformation;
        std::cout << "Transform:\n" << transformation << std::endl;

        // Apply transformation to source points
        for (auto& point : sourcePoints) {
            cv::Mat pt = (cv::Mat_<double>(3, 1) << point.x, point.y, 1.0);
            cv::Mat transformedPt = deltaTransform * pt;
            point = cv::Point2f(transformedPt.at<double>(0, 0), transformedPt.at<double>(1, 0));
            
        }
    }

    // Output final transformation matrix
    std::cout << "Final Transformation Matrix:\n" << transformation << std::endl;

    decomposeTransformation(transformation);

    std::cout << sourcePoints;
    

    return 0;
}