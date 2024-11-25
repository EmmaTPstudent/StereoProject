#include "main.h"
#include "helper.h"

using namespace cv;
using namespace std;


#include <iostream>
#include <cmath>
#include <math.h>
#include <vector>
#include <utility>
#include <queue>
#include <cstring>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;


# define pi 3.14159265358979323846  /* pi */

#define BIN_WIDTH 1                // quanti gradi all'interno della stessa categoria di voto
#define NUM_BINS 180 / BIN_WIDTH   // numero di categorie

/* === PARAMETRI PER CANNY EDGE DETECTION === */

#define KERNEL_SIZE 3
#define TRESHOLD 80
#define RATIO 3

void detectEdge(const Mat& in, Mat& out);

// FIND INTERECTION OF TWO LINES GIVEN BY POLAR COORDINATES
cv::Point2f findIntersection(const std::pair<double, double>& line1, const std::pair<double, double>& line2) {
    double rho1 = line1.first, theta1 = line1.second * pi / 180.0;
    double rho2 = line2.first, theta2 = line2.second * pi / 180.0;


    // Line equation in matrix form: A * [x, y]^T = b
    double a1 = std::cos(theta1), b1 = std::sin(theta1);
    double a2 = std::cos(theta2), b2 = std::sin(theta2);

    cv::Mat A = (cv::Mat_<double>(2, 2) << a1, b1, a2, b2);
    cv::Mat b = (cv::Mat_<double>(2, 1) << rho1, rho2);

    // Check if lines are parallel
    if (std::fabs(cv::determinant(A)) < 1e-6) {
        return cv::Point2f(-1, -1); // Return invalid point
    }

    // Solve for [x, y]
    cv::Mat intersection;
    cv::solve(A, b, intersection);

    return cv::Point2f(intersection.at<double>(0, 0), intersection.at<double>(1, 0));
}

// CHECK FOR OVERLAPPING MARKERS
bool isPointInArray(const std::vector<std::vector<cv::Point2f>>& pointsArray, const cv::Point2f& targetPoint) {
    return std::any_of(pointsArray.begin(), pointsArray.end(), [&](const std::vector<cv::Point2f>& row) {
        return std::any_of(row.begin(), row.end(), [&](const cv::Point2f& point) {
            return  abs(point.x - targetPoint.x) <= 100 && abs(point.y - targetPoint.y) <= 100; // Check for proximity
            });
        });
}

bool checkDetect(const int d, const cv::Point2f intersection, const int width, const int height, Mat img) {
    int low = 50;
    int high = 180;

    // Define rectangle corners
    Point2f p1 = intersection + Point2f(d, d);
    Point2f p2 = intersection + Point2f(d, -d);
    Point2f p3 = intersection + Point2f(-d, -d);
    Point2f p4 = intersection + Point2f(-d, d);

    // Check if all points are valid and inside the image
    if (p1.x >= 0 && p1.x < width && p1.y >= 0 && p1.y < height &&
        p2.x >= 0 && p2.x < width && p2.y >= 0 && p2.y < height &&
        p3.x >= 0 && p3.x < width && p3.y >= 0 && p3.y < height &&
        p4.x >= 0 && p4.x < width && p4.y >= 0 && p4.y < height
        ) {

        //cv::line(output, p1, p2, cv::Scalar(255, 0, 0), 1);
        //cv::line(output, p2, p3, cv::Scalar(255, 0, 0), 1);
        //cv::line(output, p3, p4, cv::Scalar(255, 0, 0), 1);
        //cv::line(output, p4, p1, cv::Scalar(255, 0, 0), 1);

        // Get intensities at the corners
        int intensity1 = img.at<uchar>(p1.y, p1.x);
        int intensity2 = img.at<uchar>(p2.y, p2.x);
        int intensity3 = img.at<uchar>(p3.y, p3.x);
        int intensity4 = img.at<uchar>(p4.y, p4.x);

        // Check for alternating intensity pattern
        if ((intensity1 < low && intensity2 > high && intensity3 < low && intensity4 > high) ||
            (intensity1 > high && intensity2 < low && intensity3 > high && intensity4 < low)) {
            //cout << "Found checkerboard at: " << endl;
            //cout << "p1: " << p1 << " p2: " << p2 << " p3:" << p3 << " p4:" << p4 << endl;
            //cout << "p1 intensity:" << intensity1 << " p2 intensity:" << intensity2 << " p3 intensity:" << intensity3 << " p4 intensity:" << intensity4 << endl;
            return 1;
        }
    }
    return 0;
}

int main() {

    string filename = "sample1_R_LedsAll_marked.png";
    int i, j;
    int theta;      // parametro di angolo di inclinazione nel sistema di coordinate polari
    double rho;     // parametro di distanza (rho) nel sistema di coordinate polari

    Mat source, edges, output;

    
    deque<pair<int, int>> edgePoints;  // <row, col>

    source = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    output = cv::imread(filename, cv::IMREAD_COLOR);
    int h = source.rows / 6;
    int w = source.cols / 6;

    imdisp("Source image", source, 0, 0, w, h, 0, 0);

    // Votes matrix: max rho, max theta
    int maxDistance = hypot(source.rows, source.cols);
    vector<vector<int>> votes(2 * maxDistance, vector<int>(NUM_BINS, 0));
    cout << "Votes matrix size: " << votes.size() << ", " << votes[0].size() << endl;

    // Detect edges using canny filter
    detectEdge(source, edges);
    imdisp("edge detection result", edges, 0, 0, w, h, 1, 0);

    // Hough tranform: Voting for the edges to detect lines
    cout << "Voting ..." << endl;
    for (i = 0; i < edges.rows; ++i) {
        for (j = 0; j < edges.cols; ++j) {
            if (edges.at<uchar>(i, j) == 255) { // Edge point
                for (theta = 0; theta < 180; theta += BIN_WIDTH) {
                    rho = round(j * cos((theta-90)*pi/180.0) + i * sin((theta - 90) * pi / 180.0)) + maxDistance;
                    if (0 < rho and (rho < (2 * maxDistance)) ) {
                        //cout << "Rho: " << rho << ", Theta: " << theta << endl;
                        votes[rho][theta]++;
                    }     
                } 
            }
        }
    }

    // Find peaks
    cout << "Finding peaks ..." << endl;
    int lineTreshold = 100;
    std::vector<std::pair<double, double>> lines; // Vector for storing the lines

    for (i = 0; i < votes.size(); ++i) {
        for (j = 0; j < votes[i].size(); ++j) {
            if (votes[i][j] >= lineTreshold) {
                rho = i - maxDistance;
                theta = j - 90;
                lines.emplace_back(rho, theta);

                // Convert polar coordinate line to two point at the image border
                double a = std::cos(theta*pi/180.0);
                double b = std::sin(theta * pi / 180.0);
                double x0 = a * rho;
                double y0 = b * rho;
                cv::Point pt1(cvRound(x0 + 10000 * (-b)), cvRound(y0 + 10000 * (a))); // Point in one direction
                cv::Point pt2(cvRound(x0 - 10000 * (-b)), cvRound(y0 - 10000 * (a))); // Point in the other direction

                // Draw the line
                cv::line(output, pt1, pt2, Scalar(255,255,255), 1);
            }
        }
    }

    imdisp("output image", output, 0, 0, w, h, 0, 1);

    std::vector<cv::Point2f> intersections;
    std::vector<std::vector<cv::Point2f>> checkerboard;
    
    // Check for checkerboard quadrants
    int width = source.cols;
    int height = source.rows;
    int d = 20; // how far from line to search check colors
    // Define axis lengths for drawing
    int axisLength = 300;
    float angle;

    // Find all intersections
    for (size_t i = 0; i < lines.size(); ++i) { // iterate through lines
        for (size_t j = i + 1; j < lines.size(); ++j) { // iterate through lines, kipping the one we already looked at

            Point2f intersection = findIntersection(lines[i], lines[j]);
            if (intersection.x >= 0 && intersection.x < source.cols && intersection.y >= 0 && intersection.y < source.rows) { // if within image

                if (not isPointInArray(checkerboard, intersection)) { // Reject intersections we already detected a check at

                    if (checkDetect(d, intersection, source.cols, source.rows, source)) {
                    
                        checkerboard.push_back({ intersection }); // add marker to list

                        circle(output, intersection, 50, Scalar(0, 255, 0), 10);

                        angle = lines[i].second * pi / 180.0;

                        // Calculate ends of axis to draw
                        cv::Point xAxisEnd(intersection.x + static_cast<int>(axisLength * cos(angle)),
                            intersection.y + static_cast<int>(axisLength * sin(angle)));
                        cv::Point yAxisEnd(intersection.x - static_cast<int>(axisLength * sin(angle)),
                            intersection.y + static_cast<int>(axisLength * cos(angle)));

                        // Draw
                        cv::line(output, intersection, xAxisEnd, cv::Scalar(0, 0, 255), 10);
                        cv::line(output, intersection, yAxisEnd, cv::Scalar(255, 0, 0), 10); 

                    }
                }
            }
        }
    }


    imdisp("Checks", output, 0, 0, w, h, 1, 1);

    //cout << checkerboard[0] << checkerboard[1] << checkerboard[2] << checkerboard[3] << endl;

    waitKey();

    return 0;
}


void detectEdge(const Mat& in, Mat& out) {
    blur(in, out, Size(3, 3));  // per immunità al rumore, sfocatura
    Canny(out, out, TRESHOLD, TRESHOLD * RATIO, KERNEL_SIZE);
}


int main1() {

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