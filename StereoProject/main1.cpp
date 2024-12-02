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
#define TRESHOLD 30
#define RATIO 3

void detectEdge(const Mat& in, Mat& out);
cv::Point2f findIntersection(const std::pair<double, double>& line1, const std::pair<double, double>& line2);
bool isPointInArray(const std::vector<cv::Point2f>& pointsArray, const cv::Point2f& targetPoint);
bool checkDetect(const int d, const cv::Point2f intersection, const float theta, const int width, const int height, Mat img, Mat& checksImg);
void drawChecks(Mat source, std::vector<std::pair<double, double>>& lines, std::vector<cv::Point2f>& checkerboard, cv::Mat& checksImg);
void houghTranform(const Mat& edgesImg, Mat& linesImg, std::vector<std::pair<double, double>>& lines);



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




int main() {



    // Images for display
    Mat source1, undistortedImg1, edgesImg1, linesImg1, checksImg1;
    // storing point cloud for each image
    std::vector<cv::Point2f> checks1;


    std::string filename1 = "gray-img01.jpg";
    Mat sourceColor1 = cv::imread(filename1, cv::IMREAD_COLOR);
    cvtColor(sourceColor1, source1, COLOR_BGR2GRAY);

    int h = source1.rows / 6;
    int w = source1.cols / 6;

    // Load the camera parameters
    float f = 3.561e3;
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << f, 0, w / 2, 0, f, h / 2, 0, 0, 1); 
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0);                


    cv::undistort(source1, undistortedImg1, cameraMatrix, distCoeffs);
    cv::undistort(sourceColor1, linesImg1, cameraMatrix, distCoeffs);

    // Detect edges using canny filter
    detectEdge(undistortedImg1, edgesImg1);
    //imdisp("Edges 1", edgesImg, 0, 0, w, h, 0, 0);

    // Hough tranform to get lit of straight lines
    std::vector<std::pair<double, double>> lines1; // Vector for storing the lines
    houghTranform(edgesImg1, linesImg1, lines1);
    checksImg1 = linesImg1.clone();
    // Detect and draw point which are at a check center
    drawChecks(undistortedImg1, lines1, checks1, checksImg1);
    imdisp("Checks 1", checksImg1, 0, 0, w, h, 0, 0);
    waitKey(1);

    // Images for display
    Mat source2, undistortedImg2, edgesImg2, linesImg2, checksImg2;
    // storing point cloud for each image
    std::vector<cv::Point2f> checks2;

    std::string filename2 = "gray-img02.jpg";
    Mat sourceColor2 = cv::imread(filename2, cv::IMREAD_COLOR);
    cvtColor(sourceColor2, source2, COLOR_BGR2GRAY);

    cv::undistort(source2, undistortedImg2, cameraMatrix, distCoeffs);
    cv::undistort(sourceColor2, linesImg2, cameraMatrix, distCoeffs);

    // Detect edges using canny filter
    detectEdge(undistortedImg2, edgesImg2);
    //imdisp("Edges 2", edgesImg, 0, 0, w, h, 1, 0);

    // Hough tranform to get lit of straight lines
    std::vector<std::pair<double, double>> lines2; // Vector for storing the lines
    houghTranform(edgesImg2, linesImg2, lines2);
    checksImg2 = linesImg2.clone();
    // Detect and draw point which are at a check center
    drawChecks(undistortedImg2, lines2, checks2, checksImg2);
    imdisp("Checks 2", checksImg2, 0, 0, w, h, 0, 1);
    waitKey(1);

    cout << "size 1: " << checks1.size() << endl;
    if (checks1.size() == 4) {
        cout << "Check 1" << checks1[0] << checks1[1] << checks1[2] << checks1[3] << endl;
    }
    else {
        cout << "Couldn't find 4 markers in image 1!" << endl;
    }

    cout << "size 2: " << checks2.size() << endl;
    if (checks2.size() == 4) {
        cout << "Check 2" << checks2[0] << checks2[1] << checks2[2] << checks2[3] << endl;
    }
    else {
        cout << "Couldn't find 4 markers in image 2!" << endl;
    }


    // Iterative Closest Point Algorithm
    cv::Mat transformation = cv::Mat::eye(3, 3, CV_64F); // Initial transformation

    for (int iter = 0; iter < 10; ++iter) {
        // Find nearest correspondences
        std::vector<int> correspondences = findCorrespondences(checks1, checks2);

        // Compute transformation
        cv::Mat deltaTransform = computeTransformation(checks1, checks2, correspondences);
        std::cout << "Delta Transform:\n" << deltaTransform << std::endl;

        // Update total transformation
        transformation = deltaTransform * transformation;
        std::cout << "Transform:\n" << transformation << std::endl;

        // Apply transformation to source points
        for (auto& point : checks1) {
            cv::Mat pt = (cv::Mat_<double>(3, 1) << point.x, point.y, 1.0);
            cv::Mat transformedPt = deltaTransform * pt;
            point = cv::Point2f(transformedPt.at<double>(0, 0), transformedPt.at<double>(1, 0));

        }
    }

    // Output final transformation matrix
    std::cout << "Final Transformation Matrix:\n" << transformation << std::endl;

    double theta, tx, ty;

    decomposeTransformation(transformation, theta, tx, ty);
    Mat undistortedTranformedImg1;

    Mat rotationMatrix = transformation(cv::Rect(0, 0, 3, 2));
    std::cout << "rotationMatrix:\n" << rotationMatrix << std::endl;

    cv::warpAffine(undistortedImg1, undistortedTranformedImg1, rotationMatrix, undistortedImg1.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

    // Blend the images with 50% transparency
    cv::Mat blendedImage;
    double alpha = 0.5; // Weight for the first image
    double beta = 1.0 - alpha; // Weight for the second image
    cv::addWeighted(undistortedImg2, alpha, undistortedTranformedImg1, beta, 0.0, blendedImage);

    // Blend the images with 50% transparency
    cv::Mat blendedImage2;
    cv::addWeighted(undistortedImg2, alpha, undistortedImg1, beta, 0.0, blendedImage2);

    imdisp("Blended image: img1, img2", blendedImage2, 0, 0, w, h, 1, 0);
    imdisp("Blended image: transformed img 1, img2", blendedImage, 0, 0, w, h, 1, 1);
    waitKey(1);


    // Setup a rectangle to define your region of interest
    Rect ROI(1300,600,2100,1600);
    Mat left = undistortedImg2.clone();
    Mat right = undistortedTranformedImg1.clone();
    left = left(ROI);
    right = right(ROI);

    cv::equalizeHist(left, left);
    cv::equalizeHist(right, right);
    cv::GaussianBlur(left, left, cv::Size(5, 5), 0);
    cv::GaussianBlur(right, right, cv::Size(5, 5), 0);

    int ch = left.rows / 6;
    int cw = left.cols / 6;

    imdisp("Cropped image 1", left, 0, 0,cw,ch, 0, 0);
    imdisp("Cropped image 2",right, 0, 0, cw, ch, 1, 0);

    waitKey(1);
    cout << "Stereo..." << endl;

    // Parameters for StereoSGBM
    int numDisparities = 16 * 10;// Must be divisible by 16
    int blockSize = 19;      // Block size to match

    // Create StereoSGBM object
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(
        0,                        // Min disparity
        numDisparities,           // Max disparity range
        blockSize                 // Block size
    );

    stereo->setP1(8 * left.channels() * blockSize * blockSize); // Smoothness penalty
    stereo->setP2(32 * left.channels() * blockSize * blockSize); // Smoothness penalty
    stereo->setPreFilterCap(15);
    stereo->setUniquenessRatio(20); // Ratio to filter ambiguous matches
    stereo->setSpeckleWindowSize(100);
    stereo->setSpeckleRange(32);
    stereo->setDisp12MaxDiff(1);

    // Compute disparity map
    cv::Mat disparity;
    stereo->compute(left, right, disparity);

    cv::medianBlur(disparity, disparity, 5);

    // Normalize the disparity map for visualization
    cv::Mat dispVis;
    cv::normalize(disparity, dispVis, 0, 255, cv::NORM_MINMAX, CV_8U);

    imdisp("Diparity map", dispVis, 0, 0, cw * 2, ch * 2, 1, 0);

    waitKey(0);
    waitKey(0);
    waitKey(0);
    return 0;
}









// FUNCTIONS ////////////////////////////////////////////////////////////////////////////////////////////

void detectEdge(const Mat& in, Mat& out) {
    blur(in, out, Size(3, 3));  // per immunità al rumore, sfocatura
    Canny(out, out, TRESHOLD, TRESHOLD * RATIO, KERNEL_SIZE);
}


// FIND INTERECTION OF TWO LINES GIVEN BY POLAR COORDINATES
cv::Point2f findIntersection(const std::pair<double, double>&line1, const std::pair<double, double>&line2) {
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
bool isPointInArray(const std::vector<cv::Point2f>& pointsArray, const cv::Point2f& targetPoint) {
    return std::any_of(pointsArray.begin(), pointsArray.end(), [&](const cv::Point2f& point) {
        return  abs(point.x - targetPoint.x) <= 100 && abs(point.y - targetPoint.y) <= 100; // Check for proximity

    });
}

// DETECT A 2X2 CHECKERBOARD PATTERN AROUND GIVEN POINT
bool checkDetect(const int d, const cv::Point2f intersection, float theta, const int width, const int height, Mat img, Mat&checksImg) {
    int low = 60;
    int high = 90;

    // Precompute rotation values
    float sinTheta = sin(theta * pi / 180);
    float cosTheta = cos(theta * pi / 180);

    // Define rectangle corners with rotation
    Point2f p1(Point2f(intersection.x + d * cosTheta - d * sinTheta, intersection.y + d * cosTheta + d * sinTheta));
    Point2f p2(Point2f(intersection.x + d * cosTheta + d * sinTheta, intersection.y + -d * cosTheta + d * sinTheta));
    Point2f p3(Point2f(intersection.x + -d * cosTheta + d * sinTheta, intersection.y + -d * cosTheta - d * sinTheta));
    Point2f p4(Point2f(intersection.x + -d * cosTheta - d * sinTheta, intersection.y + d * cosTheta - d * sinTheta));


    // Check if all points are valid and inside the image
    if (p1.x >= 0 && p1.x < width && p1.y >= 0 && p1.y < height &&
        p2.x >= 0 && p2.x < width && p2.y >= 0 && p2.y < height &&
        p3.x >= 0 && p3.x < width && p3.y >= 0 && p3.y < height &&
        p4.x >= 0 && p4.x < width && p4.y >= 0 && p4.y < height
        ) {

        // Get intensities at the corners
        int intensity1 = img.at<uchar>(p1.y, p1.x);
        int intensity2 = img.at<uchar>(p2.y, p2.x);
        int intensity3 = img.at<uchar>(p3.y, p3.x);
        int intensity4 = img.at<uchar>(p4.y, p4.x);

        // Check for alternating intensity pattern
        if ((intensity1 < low && intensity2 > high && intensity3 < low && intensity4 > high) ||
            (intensity1 > high && intensity2 < low && intensity3 > high && intensity4 < low)) {
            cv::line(checksImg, p1, p2, cv::Scalar(0, 255, 0), 2);
            cv::line(checksImg, p2, p3, cv::Scalar(0, 255, 0), 2);
            cv::line(checksImg, p3, p4, cv::Scalar(0, 255, 0), 2);
            cv::line(checksImg, p4, p1, cv::Scalar(0, 255, 0), 2);
            return 1;

        }
    }
    return 0;
}


void drawChecks(Mat source, std::vector<std::pair<double, double>>& lines, std::vector<cv::Point2f>& checkerboard, Mat& checksImg) {
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

                    if (checkDetect(d, intersection, lines[i].second, source.cols, source.rows, source, checksImg)) {

                        checkerboard.push_back({ intersection }); // add marker to list

                        circle(checksImg, intersection, 50, Scalar(0, 255, 0), 10);

                        angle = lines[i].second * pi / 180.0;

                        // Calculate ends of axis to draw
                        cv::Point xAxisEnd(intersection.x + static_cast<int>(axisLength * cos(angle)),
                            intersection.y + static_cast<int>(axisLength * sin(angle)));
                        cv::Point yAxisEnd(intersection.x - static_cast<int>(axisLength * sin(angle)),
                            intersection.y + static_cast<int>(axisLength * cos(angle)));

                        // Draw
                        cv::line(checksImg, intersection, xAxisEnd, cv::Scalar(0, 0, 255), 10);
                        cv::line(checksImg, intersection, yAxisEnd, cv::Scalar(255, 0, 0), 10);

                    }
                }
            }
        }
    }
}

void houghTranform(const Mat& edgesImg, Mat& linesImg, std::vector<std::pair<double, double>>& lines) {
    // Hough tranform: Voting for the edges to detect lines
    int maxDistance = hypot(edgesImg.rows, edgesImg.cols);
    vector<vector<int>> votes(2 * maxDistance, vector<int>(NUM_BINS, 0));

    for (int i = 0; i < edgesImg.rows; ++i) {
        for (int j = 0; j < edgesImg.cols; ++j) {
            if (edgesImg.at<uchar>(i, j) == 255) { // Edge point are white
                for (double theta = 0; theta < 180; theta += BIN_WIDTH) {
                    double rho = round(j * cos((theta - 90) * pi / 180.0) + i * sin((theta - 90) * pi / 180.0)) + maxDistance;
                    if (0 < rho and (rho < (2 * maxDistance))) {
                        //cout << "Rho: " << rho << ", Theta: " << theta << endl;
                        votes[rho][theta]++;
                    }
                }
            }
        }
    }

    // Find peaks
    int lineTreshold = 100;

    double rho, theta;

    for (int i = 0; i < votes.size(); ++i) {
        for (int j = 0; j < votes[i].size(); ++j) {
            if (votes[i][j] >= lineTreshold) {
                rho = i - maxDistance;
                theta = j - 90;
                lines.emplace_back(rho, theta);

                // Convert polar coordinate line to two point at the image border
                double a = std::cos(theta * pi / 180.0);
                double b = std::sin(theta * pi / 180.0);
                double x0 = a * rho;
                double y0 = b * rho;
                cv::Point pt1(cvRound(x0 + 10000 * (-b)), cvRound(y0 + 10000 * (a))); // Point in one direction
                cv::Point pt2(cvRound(x0 - 10000 * (-b)), cvRound(y0 - 10000 * (a))); // Point in the other direction

                // Draw the line
                cv::line(linesImg, pt1, pt2, Scalar(255, 255, 255), 1);
            }
        }
    }
}
 