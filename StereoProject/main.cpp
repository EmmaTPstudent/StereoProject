#include "main.h"
#include "helper.h"

using namespace cv;
using namespace std;


int main() {

    // Load the stereo images
    cv::Mat img1 = cv::imread("sample1_L_LedsAll.jpg", cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread("sample1_R_LedsAll.jpg", cv::IMREAD_COLOR);

    // Check if the images are loaded correctly
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error loading images!" << std::endl;
        return -1;
    }

    int w = img1.cols;
    int h = img1.rows;
    int dispw = w / 6;
    int disph = h / 6;

    imdisp("Left", img1, 10, 10, dispw, disph, 0, 0);
    imdisp("Right", img2, 10, 10, dispw, disph, 1, 0);

    //waitKey(0);


    float f = 4.74*1000;
    float d = 1.4 * 0.001; //sensor_width / image_width = physical size of pixel


    // Load the camera parameters
    cv::Mat cameraMatrix1 = (cv::Mat_<double>(3, 3) << f, 0, w/2, 0, f, h/2, 0, 0, 1); // Example values
    cv::Mat distCoeffs1 = (cv::Mat_<double>(1, 5) << 0.1, -0.05, 0, 0, 0);                   // Example values
    cv::Mat cameraMatrix2 = (cv::Mat_<double>(3, 3) << f, 0, w/2, 0, f, h/2, 0, 0, 1); // Example values
    cv::Mat distCoeffs2 = (cv::Mat_<double>(1, 5) << 0.1, -0.05, 0, 0, 0);                   // Example values



    // Known rotation matrix and translation vector between the cameras
    Mat R = eulerToRotationMatrix(0, 0, 0.05);


    cv::Mat T = (cv::Mat_<double>(3, 1) << 1, 0.0, 0.0);   
    
    // Image size
    cv::Size imageSize(w, h); // Example image size

    // Rectification matrices
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, R1, R2, P1, P2, Q);

    // Compute rectification maps for both cameras
    cv::Mat map1x, map1y, map2x, map2y;
    cv::initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, CV_32FC1, map1x, map1y);
    cv::initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, CV_32FC1, map2x, map2y);


    // Rectify the images
    cv::Mat rectifiedImage1, rectifiedImage2;
    cv::remap(img1, rectifiedImage1, map1x, map1y, cv::INTER_LINEAR);
    cv::remap(img2, rectifiedImage2, map2x, map2y, cv::INTER_LINEAR);

    // Display the rectified images
    imdisp("rectifiedImage1 Left", rectifiedImage1, 10, 10, dispw, disph, 0, 1);
    imdisp("rectifiedImage2 Right", rectifiedImage2, 10, 10, dispw, disph, 1, 1);
    cv::waitKey(0);

    return 0;



} 