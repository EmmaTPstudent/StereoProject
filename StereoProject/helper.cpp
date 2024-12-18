// Author: Emma Therese porsbjerg, s184751


#include "helper.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;


int imdisp(const std::string& title, const cv::Mat& image, const int& sx, const int& sy, const int& w, const int& h, const int& x, const int& y) {
    // Displays one image at the given coordinates x,y, in a grid defined by width and height. The grid starts at sx, sy
    // if given width and height is different form actual image, image will be stretched to fit
    cv::Mat imageResized;
    cv::resize(image, imageResized, cv::Size(w, h), 0, 0, cv::INTER_AREA);
    cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(title, sx + x * w, sy + y * (h + 30));
    cv::imshow(title, imageResized);
    cv::resizeWindow(title, w, h);
    return 0;
}


bool loadTwoImages(const std::string& folderPath, cv::Mat& img1, cv::Mat& img2) {
    std::vector<std::string> imagePaths;

    // Iterate through the folder to collect image paths
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();

            // Check for valid image extensions
            if (filePath.ends_with(".jpg") || filePath.ends_with(".png") || filePath.ends_with(".bmp")) {
                imagePaths.push_back(filePath);
            }
        }
    }

    // Check if exactly two images were found
    if (imagePaths.size() != 2) {
        std::cerr << "Error: The folder must contain exactly two image files. Found " << imagePaths.size() << " file(s)." << std::endl;
        return false;
    }

    // Load the two images into the cv::Mat objects
    img1 = cv::imread(imagePaths[0], cv::IMREAD_COLOR);
    img2 = cv::imread(imagePaths[1], cv::IMREAD_COLOR);

    // Check if images were loaded successfully
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Failed to load one or both images." << std::endl;
        return false;
    }

    return true;
}





// Function to convert Euler angles (in radians) to a rotation matrix
cv::Mat eulerToRotationMatrix(double roll, double pitch, double yaw) {
    // Compute the individual rotation matrices
    cv::Mat Rx = (cv::Mat_<double>(3, 3) <<
        1, 0, 0,
        0, cos(roll), -sin(roll),
        0, sin(roll), cos(roll));

    cv::Mat Ry = (cv::Mat_<double>(3, 3) <<
        cos(pitch), 0, sin(pitch),
        0, 1, 0,
        -sin(pitch), 0, cos(pitch));

    cv::Mat Rz = (cv::Mat_<double>(3, 3) <<
        cos(yaw), -sin(yaw), 0,
        sin(yaw), cos(yaw), 0,
        0, 0, 1);

    // The combined rotation matrix: R = Rz * Ry * Rx
    cv::Mat R = Rz * Ry * Rx;

    return R;
}


void decomposeTransformation(const cv::Mat& transformation, double& rotationAngle, double& tx, double& ty) {
    // Extract translation vector
    tx = transformation.at<double>(0, 2);
    ty = transformation.at<double>(1, 2);
    std::cout << "Translation Vector:\n";
    std::cout << "[ " << tx << ", " << ty << " ]" << std::endl;
    // Extract rotation matrix elements
    double r11 = transformation.at<double>(0, 0);
    double r21 = transformation.at<double>(1, 0);
    // Compute the rotation angle (in radians)
    rotationAngle = std::atan2(r21, r11);
    std::cout << "Rotation Angle (in degrees): " << rotationAngle * (180.0 / CV_PI) << std::endl;
}


// Euclidean distance between two cv::Point2f
double euclideanDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2)); // d_e = sqrt(d_x^2 + d_y^2)
}


std::pair<cv::Point2f, cv::Point2f> findClosestPair(const std::vector<cv::Point2f>& points, float& minDistance) {
    // Initialize the minimum distance with the maximum possible value
    minDistance = std::numeric_limits<float>::max();
    std::pair<cv::Point2f, cv::Point2f> closestPair;

    // Iterate through all pairs of points
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            // Calculate the Euclidean distance
            float distance = euclideanDistance(points[i], points[j]);

            // Update minimum distance and closest pair
            if (distance < minDistance) {
                minDistance = distance;
                closestPair = { points[i], points[j] };
            }
        }
    }

    return closestPair;
}