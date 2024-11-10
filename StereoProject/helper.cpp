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
    cv::resize(image, imageResized, cv::Size(), w / image.cols, h / image.rows);
    cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(title, sx + x * w, sy + y * (h + 30));
    cv::imshow(title, imageResized);
    cv::resizeWindow(title, w, h);
    return 0;
}