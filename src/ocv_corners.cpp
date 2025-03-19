#include "ocv_corners.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;


// Funzione per ridimensionare l'immagine sfruttando l'interpolazione bilineare
void resizeImage(const Mat& src, Mat& dst, int newWidth, int newHeight) {
    Size newSize(newWidth, newHeight);
    resize(src, dst, newSize, 0, 0, INTER_LINEAR);
}

void detectShiTomasiCorners(const Mat& src, Mat& dst, vector<Point2f>& corners, int maxCorners, double qualityLevel, double minDistance) {
    // Converto l'immagine in scala di grigi
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    // Trovo i corner con Shi-Tomasi
    goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance, Mat(), 3, false, 0.04);

    // Disegnamo i corner appena rilevati
    dst = src.clone();
    for (size_t i = 0; i < corners.size(); i++) {
        circle(dst, corners[i], 3, Scalar(0, 255, 0), FILLED);
    }
}

void detectFASTCorners(const Mat& src, Mat& dst, vector<KeyPoint>& keypoints, int threshold, bool nonMaxSuppression) {
    // Converto in scala di grigi
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    // Creo il detector FAST
    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(threshold, nonMaxSuppression);
    fast->detect(gray, keypoints);

    // Disegno i keypoints sull'immagine
    dst = src.clone();
    drawKeypoints(src, keypoints, dst, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

