#include "ocv_corners.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Funzione per rilevare gli angoli Shi-Tomasi
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

// Funzione per rilevare i corner con il metodo FAST
void detectFASTCorners(const Mat& src, Mat& dst, vector<KeyPoint>& keypoints, int threshold, bool nonMaxSuppression) {
    // Converto in scala di grigi
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    // Creo il detector FAST
    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(threshold, nonMaxSuppression);
    fast->detect(gray, keypoints);

    // Disegno i keypoints sull'immagine
    dst = src.clone();
    for (const auto& kp : keypoints) {
        Point pt(cvRound(kp.pt.x), cvRound(kp.pt.y));
        circle(dst, pt, 3, Scalar(0, 255, 0), FILLED);
    }
}

// Funzione per rilevare i corner con il metodo Harris
void detectHarrisCorners(const Mat& src, Mat& dst, double blockSize, double ksize, double k,
    double threshold, double minDistance, vector<KeyPoint>& keypoints) {
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    Mat harrisResponse;
    cornerHarris(gray, harrisResponse, static_cast<int>(blockSize), static_cast<int>(ksize), k);

    Mat harrisNorm;
    normalize(harrisResponse, harrisNorm, 0, 255, NORM_MINMAX, CV_32FC1);

    dst = src.clone();
    keypoints.clear();

    // Per salvare i punti che rispettano la distanza minima
    vector<Point2f> acceptedCorners;

    for (int y = 0; y < harrisNorm.rows; y++) {
        for (int x = 0; x < harrisNorm.cols; x++) {
            float response = harrisNorm.at<float>(y, x);
            if (response > threshold) {
                Point2f pt(x, y);

                // Verifico se è lontano abbastanza dagli altri corner
                bool tooClose = false;
                for (const auto& existing : acceptedCorners) {
                    if (norm(existing - pt) < minDistance) {
                        tooClose = true;
                        break;
                    }
                }

                if (!tooClose) {
                    acceptedCorners.push_back(pt);
                    keypoints.emplace_back(KeyPoint(pt, 1.f));
                    circle(dst, pt, 3, Scalar(0, 255, 0), FILLED);
                }
            }
        }
    }
}



