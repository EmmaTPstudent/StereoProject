#pragma once

// Author: Emma Therese porsbjerg, s184751
#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>


// Function declaration (prototype)
int imdisp(const std::string& title, const cv::Mat& image, const int& sx, const int& sy, const int& w, const int& h, const int& x, const int& y);
bool loadTwoImages(const std::string& folderPath, cv::Mat& img1, cv::Mat& img2);


cv::Mat eulerToRotationMatrix(double roll, double pitch, double yaw);
void decomposeTransformation(const cv::Mat& transformation, double& rotationAngle, double& tx, double& ty);
double euclideanDistance(const cv::Point2f& p1, const cv::Point2f& p2);

std::pair<cv::Point2f, cv::Point2f> findClosestPair(const std::vector<cv::Point2f>& points, float& minDistance);




#endif // HELPER_H