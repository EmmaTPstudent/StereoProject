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

void replaceInvalidWithNearest(cv::Mat& image, int lowerBound, int upperBound);

void detectEdge(const Mat& in, Mat& out);
cv::Point2f findIntersection(const std::pair<double, double>& line1, const std::pair<double, double>& line2);
bool isPointInArray(const std::vector<cv::Point2f>& pointsArray, const cv::Point2f& targetPoint);
bool checkDetect(const int d, const cv::Point2f intersection, const float theta, Mat img, Mat& checksImg);
void houghTranform(const Mat& edgesImg, Mat& linesImg, std::vector<std::pair<double, double>>& lines);
bool getMarkers(cv::Mat input, cv::Mat cameraMatrix, cv::Mat distCoeffs, cv::Mat& output, std::vector<cv::Point2f>& cloud, float& minDistance);





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








int main() {

    // Load raw images
    std::string folderPath = "./pair";
    Mat source1, source2, imgColor1, img1, imgColor2, img2;
    loadTwoImages(folderPath, source1, source2);

    int h = source1.rows / 10; // height and width used for display
    int w = source2.cols / 10;

    imdisp("source1", source1, 0, 0, w, h, 0, 0);
    imdisp("source2", source2, 0, 0, w, h, 0, 1);
    waitKey(1);

    // Load the camera parameters for un-warping
    float f = 3.561e3;
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << f, 0, w / 2, 0, f, h / 2, 0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0);

    Mat markersImg1, markersImg2;
    std::vector<cv::Point2f> cloud1, cloud2;
    float minDist;

    cv::undistort(source1, imgColor1, cameraMatrix, distCoeffs);
    cvtColor(imgColor1, img1, COLOR_BGR2GRAY);
    getMarkers(imgColor1, cameraMatrix, distCoeffs, markersImg1, cloud1, minDist);
    imdisp("markersImg1", markersImg1, 0, 0, w, h, 0, 0);
    waitKey(1);

    cv::undistort(source2, imgColor2, cameraMatrix, distCoeffs);
    cvtColor(imgColor2, img2, COLOR_BGR2GRAY);
    getMarkers(imgColor2, cameraMatrix, distCoeffs, markersImg2, cloud2, minDist);
    imdisp("markersImg2", markersImg2, 0, 0, w, h, 0, 1);
    waitKey(1);




    // ICP - Iterative Closest Point Algorithm /////////////////////////////////////////////////////////////
    cv::Mat transformation = cv::Mat::eye(3, 3, CV_64F); // Initial transformation

    for (int iter = 0; iter < 10; ++iter) {
        // Find nearest correspondences
        std::vector<int> correspondences = findCorrespondences(cloud1, cloud2);

        // Compute transformation
        cv::Mat deltaTransform = computeTransformation(cloud1, cloud2, correspondences);
        std::cout << "Delta Transform:\n" << deltaTransform << std::endl;

        // Update total transformation
        transformation = deltaTransform * transformation;
        std::cout << "Transform:\n" << transformation << std::endl;

    }

    std::cout << "Final Transformation Matrix:\n" << transformation << std::endl;


    double theta, tx, ty;

    decomposeTransformation(transformation, theta, tx, ty);
    
    Mat rotationMatrix = transformation(cv::Rect(0, 0, 3, 2));
    std::cout << "rotationMatrix:\n" << rotationMatrix << std::endl;

    Mat transformedImg1;
    cv::warpAffine(img1, transformedImg1, rotationMatrix, img1.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));



    float realDist = 2.752; // measured with calipers
    float S = realDist / minDist;

    float baseline = S * sqrt(pow(tx, 2) + pow(ty, 2));
    std::cout << "Physical baseline: " << baseline << std::endl;






    // Blend the images with 50% transparency
    cv::Mat blendedImage;
    double alpha = 0.5; // Weight for the first image
    double beta = 1.0 - alpha; // Weight for the second image
    cv::addWeighted(img2, alpha, transformedImg1, beta, 0.0, blendedImage);

    // Blend the images with 50% transparency
    cv::Mat blendedImage2;
    cv::addWeighted(img2, alpha, img1, beta, 0.0, blendedImage2);

    imdisp("Blended image: img1, img2", blendedImage2, 0, 0, w, h, 1, 0);
    imdisp("Blended image: transformed img1, img2", blendedImage, 0, 0, w, h, 1, 1);
    waitKey(1);


    // Setup a rectangle to define your region of interest
    Rect ROI(1600,650,1400,1400); //px, py, w, h

    Mat left = img2.clone();
    Mat right = transformedImg1.clone();
    
    left = left(ROI);
    right = right(ROI);

    cv::equalizeHist(left, left);
    cv::equalizeHist(right, right);
    cv::GaussianBlur(left, left, cv::Size(5, 5), 0);
    cv::GaussianBlur(right, right, cv::Size(5, 5), 0);

    int ch = left.rows / 3;
    int cw = left.cols / 3;

    imdisp("Cropped image 1", left, 0, 0, cw, ch, 0, 0);
    imdisp("Cropped image 2", right, 0, 0, cw, ch, 1, 0);

    waitKey(1);
    cout << "Stereo..." << endl;

    // Parameters for StereoSGBM
    int numDisparities = 16 * 10;// Must be divisible by 16
    int blockSize = 11;      // Block size to match

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

    imdisp("Disparity map", dispVis, 0, 0, cw, ch, 0, 1);
    waitKey(1);

    Mat fixedDisp = dispVis.clone();
    // Define intensity range
    int lowerBound = 80;
    int upperBound = 130;

    // Replace invalid pixels with the nearest valid neighbor
    replaceInvalidWithNearest(fixedDisp, lowerBound, upperBound);

    imdisp("Disparity map, flooded", fixedDisp, 0, 0, cw, ch, 2, 1);
    waitKey(1); 

    //Rect ROI2(500, 300, 800, 900);
    Mat fixedDispMinMax = fixedDisp.clone();
    //fixedDispMinMax = fixedDispMinMax(ROI2);

    cv::normalize(fixedDispMinMax, fixedDispMinMax, 0, 255, cv::NORM_MINMAX, CV_8U);

    imdisp("Disparity map, flooded, minMax", fixedDispMinMax, 0, 0, cw, ch, 3, 1);
    waitKey(1);



    waitKey(0);
    waitKey(0);
    waitKey(0);
    return 0;
}




// FUNCTIONS ////////////////////////////////////////////////////////////////////////////////////////////



bool getMarkers(cv::Mat input, cv::Mat cameraMatrix, cv::Mat distCoeffs, cv::Mat& output, std::vector<cv::Point2f>& cloud, float& minDistance) {
    Mat inputGray, edgesImg, checksImg;
    cvtColor(input, inputGray, COLOR_BGR2GRAY);

    // Detect edges using canny filter
    detectEdge(inputGray, edgesImg);

    // Hough tranform to get list of straight lines
    std::vector<std::pair<double, double>> lines; // Vector for storing the lines
    Mat markerImg = input.clone();
    houghTranform(edgesImg, markerImg, lines);

    // Find all intersections
    int d = 10; // how far from line to search check colors
    // Define axis lengths for drawing
    int axisLength = 300;
    float angle;
    for (size_t i = 0; i < lines.size(); ++i) { // iterate through lines
        for (size_t j = i + 1; j < lines.size(); ++j) { // iterate through lines, kipping the one we already looked at
            Point2f intersection = findIntersection(lines[i], lines[j]);
            if (intersection.x >= 0 && intersection.x < input.cols && intersection.y >= 0 && intersection.y < input.rows) { // if within image

                if (not isPointInArray(cloud, intersection)) { // Reject intersections we already detected a check at

                    if (checkDetect(d, intersection, lines[i].second, inputGray, markerImg)) {

                        cloud.push_back({ intersection }); // add marker to list

                        circle(markerImg, intersection, 50, Scalar(0, 255, 0), 10);

                        angle = lines[i].second * pi / 180.0;

                        // Calculate ends of axis to draw
                        cv::Point xAxisEnd(intersection.x + static_cast<int>(axisLength * cos(angle)),
                            intersection.y + static_cast<int>(axisLength * sin(angle)));
                        cv::Point yAxisEnd(intersection.x - static_cast<int>(axisLength * sin(angle)),
                            intersection.y + static_cast<int>(axisLength * cos(angle)));

                        // Draw
                        cv::line(markerImg, intersection, xAxisEnd, cv::Scalar(0, 0, 255), 10);
                        cv::line(markerImg, intersection, yAxisEnd, cv::Scalar(255, 0, 0), 10);
                    }
                }
            }
        }
    }

    cout << "No. of markers found: " << cloud.size() << endl;
    if (cloud.size() == 4) {
        cout << "Cloud: " << cloud[0] << cloud[1] << cloud[2] << cloud[3] << endl;
    }
    else {
        cout << "Error! Needs 4 markers exactly" << endl;
    }

    // Getting the minimum distance is used to calculate the baseline (physical distance is known)
    auto closestPair = findClosestPair(cloud, minDistance);
    std::cout << "Closest Pair: (" << closestPair.first.x << ", " << closestPair.first.y << ") and ("
        << closestPair.second.x << ", " << closestPair.second.y << ")" << std::endl;
    std::cout << "Minimum Distance: " << minDistance << std::endl;


    output = markerImg;

    return 1;


}




// Replace all pixels outside the given range with the nearest valid neighbor
void replaceInvalidWithNearest(cv::Mat& image, int lowerBound, int upperBound) {
    if (image.empty()) {
        std::cerr << "Error: Input image is empty!" << std::endl;
        return;
    }

    // Create a mask to mark valid and invalid pixels
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);

    // Check if the image is grayscale or color
    if (image.channels() == 1) {
        // Grayscale image
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                uchar pixel = image.at<uchar>(y, x);
                if (pixel >= lowerBound && pixel <= upperBound) {
                    mask.at<uchar>(y, x) = 255; // Mark valid pixels
                }
            }
        }
    }
    else if (image.channels() == 3) {
        // Color image
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
                if (pixel[0] >= lowerBound && pixel[0] <= upperBound &&
                    pixel[1] >= lowerBound && pixel[1] <= upperBound &&
                    pixel[2] >= lowerBound && pixel[2] <= upperBound) {
                    mask.at<uchar>(y, x) = 255; // Mark valid pixels
                }
            }
        }
    }
    else {
        std::cerr << "Unsupported number of channels: " << image.channels() << std::endl;
        return;
    }

    // Inpaint invalid regions using the nearest valid pixel
    if (image.channels() == 1) {
        // Grayscale image
        cv::inpaint(image, 255 - mask, image, 3, cv::INPAINT_TELEA);
    }
    else if (image.channels() == 3) {
        // Color image
        cv::Mat inpaintedImage;
        cv::inpaint(image, 255 - mask, inpaintedImage, 3, cv::INPAINT_TELEA);
        image = inpaintedImage;
    }
}





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
bool checkDetect(const int d, const cv::Point2f intersection, float theta, Mat img, Mat& checksImg) {
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
    if (p1.x >= 0 && p1.x < img.cols && p1.y >= 0 && p1.y < img.rows &&
        p2.x >= 0 && p2.x < img.cols && p2.y >= 0 && p2.y < img.rows &&
        p3.x >= 0 && p3.x < img.cols && p3.y >= 0 && p3.y < img.rows &&
        p4.x >= 0 && p4.x < img.cols && p4.y >= 0 && p4.y < img.rows
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
 