#include "main.h"
#include "helper.h"

using namespace cv;
using namespace std;


int main() {

    // Read image
    string filename = "C:/Users/EmmaP/OneDrive/MSc/E24/Image Analysis on Microcontrollers/bar2.png";
    Mat image = imread(filename, IMREAD_GRAYSCALE);

    // Check if the image is loaded
    if (image.empty()) {
        std::cerr << "Error: Could not load image: " << filename << std::endl;
        return 1;
    }

    int w = image.cols;
    int h = image.rows;

    imdisp("Thresholded image", image, 400, 50, w, h, 0, 0);

    waitKey(0);
}