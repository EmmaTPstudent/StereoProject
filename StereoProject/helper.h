#pragma once

// Author: Emma Therese porsbjerg, s184751
#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>


// Function declaration (prototype)
int imdisp(const std::string& title, const cv::Mat& image, const int& sx, const int& sy, const int& w, const int& h, const int& x, const int& y);
cv::Mat eulerToRotationMatrix(double roll, double pitch, double yaw);

#endif // HELPER_H