// Author: Emma Therese porsbjerg, s184751


#include "helper.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


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