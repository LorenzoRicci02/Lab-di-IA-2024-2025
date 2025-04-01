#include "my_corners.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// Shi-Tomasi

pair<Mat, Mat> computeDerivatives(const Mat& image) {
    Mat gradX, gradY;
    Sobel(image, gradX, CV_64F, 1, 0, 3);
    Sobel(image, gradY, CV_64F, 0, 1, 3);
    return {gradX, gradY};
}

Matx22d buildStructureMatrix(int posX, int posY, const Mat& gradX, const Mat& gradY, int kernelSize) {
    double sumXX = 0, sumYY = 0, sumXY = 0;
    int halfSize = kernelSize / 2;

    for (int offsetY = -halfSize; offsetY <= halfSize; offsetY++) {
        for (int offsetX = -halfSize; offsetX <= halfSize; offsetX++) {
            double dx = gradX.at<double>(posY + offsetY, posX + offsetX);
            double dy = gradY.at<double>(posY + offsetY, posX + offsetX);
            sumXX += dx * dx;
            sumYY += dy * dy;
            sumXY += dx * dy;
        }
    }

    return Matx22d(sumXX, sumXY, sumXY, sumYY);
}

vector<double> extractEigenvalues(const Matx22d& structMatrix) {
    Mat eigenVals;
    eigen(structMatrix, eigenVals);
    return {eigenVals.at<double>(0), eigenVals.at<double>(1)};
}

Mat nonMaximumSuppression(const Mat& responseMap, int windowSize) {
    Mat filtered = Mat::zeros(responseMap.size(), CV_64F);
    int halfWindow = windowSize / 2;

    for (int row = halfWindow; row < responseMap.rows - halfWindow; row++) {
        for (int col = halfWindow; col < responseMap.cols - halfWindow; col++) {
            double currentValue = responseMap.at<double>(row, col);
            bool isMaximum = true;

            for (int offsetY = -1; offsetY <= 1; offsetY++) {
                for (int offsetX = -1; offsetX <= 1; offsetX++) {
                    if (responseMap.at<double>(row + offsetY, col + offsetX) > currentValue) {
                        isMaximum = false;
                        break;
                    }
                }
                if (!isMaximum) break;
            }

            if (isMaximum) {
                filtered.at<double>(row, col) = currentValue;
            }
        }
    }

    return filtered;
}

vector<KeyPoint> ShiTomasiCorners(const Mat& sourceImage, double qualityRatio, int maxPoints, int patchSize) {
    Mat grayImg;
    cvtColor(sourceImage, grayImg, COLOR_BGR2GRAY);
    grayImg.convertTo(grayImg, CV_64F);

    auto [gradX, gradY] = computeDerivatives(grayImg);
    Mat scoreMap = Mat::zeros(grayImg.size(), CV_64F);
    double peakResponse = 0;

    int halfPatch = patchSize / 2;
    for (int row = halfPatch; row < grayImg.rows - halfPatch; row++) {
        for (int col = halfPatch; col < grayImg.cols - halfPatch; col++) {
            Matx22d localMatrix = buildStructureMatrix(col, row, gradX, gradY, patchSize);
            auto eigenVals = extractEigenvalues(localMatrix);
            double minEigenvalue = min(eigenVals[0], eigenVals[1]);

            scoreMap.at<double>(row, col) = minEigenvalue;
            if (minEigenvalue > peakResponse) peakResponse = minEigenvalue;
        }
    }

    double scoreThreshold = qualityRatio * peakResponse;
    Mat prunedCorners = nonMaximumSuppression(scoreMap, patchSize);

    // Creazione della lista di keypoint
    vector<KeyPoint> detectedPoints;
    for (int row = 0; row < prunedCorners.rows; row++) {
        for (int col = 0; col < prunedCorners.cols; col++) {
            if (prunedCorners.at<double>(row, col) > scoreThreshold) {
                detectedPoints.push_back(KeyPoint(col, row, 1));
            }
        }
    }

    if (detectedPoints.size() > maxPoints) {
        sort(detectedPoints.begin(), detectedPoints.end(), [](const KeyPoint& a, const KeyPoint& b) {
            return a.response > b.response;
        });
        detectedPoints.resize(maxPoints);
    }

    return detectedPoints;
}

// FAST

vector<Point> circonferenza = {
    {0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
    {0, 3}, {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}
};

bool fastSegmentTest(const Mat &img, int x, int y, int threshold, int n) {
    uchar pixel = img.at<uchar>(y, x);
    int countHigh = 0, countLow = 0;
    for (int i = 0; i < 16; i++) {
        int dx = x + circonferenza[i].x;
        int dy = y + circonferenza[i].y;
        if (dx < 0 || dx >= img.cols || dy < 0 || dy >= img.rows)
            continue;
        uchar nearest = img.at<uchar>(dy, dx);
        if (nearest > pixel + threshold) {
            countHigh++;
        } else if (nearest < pixel - threshold) {
            countLow++;
        }
    }
    return (countHigh >= n || countLow >= n);
}

bool fastHighSpeedTest(const Mat &img, int x, int y, int threshold) {
    uchar pixel = img.at<uchar>(y, x);
    vector<int> test_pixel = {1, 9, 5, 13};
    int countHigh = 0, countLow = 0;
    for (int i = 0; i < test_pixel.size(); i++) {
        int j = test_pixel[i];
        int dx = x + (circonferenza[j].x);
        int dy = y + (circonferenza[j].y);
        if (dx < 0 || dx >= img.cols || dy < 0 || dy >= img.rows)
            continue;
        uchar nearest = img.at<uchar>(dy, dx);
        if (nearest > pixel + threshold) {
            countHigh++;
        } else if (nearest < pixel - threshold) {
            countLow++;
        }
    }
    return (countHigh >= 3 || countLow >= 3);
}

void nonMaximumSuppression(vector<KeyPoint> &keypoints, const Mat &img, int dist) {
    vector<float> V_keypoints(keypoints.size());
    for (int i = 0; i < keypoints.size(); i++) {
        KeyPoint kp = keypoints[i];
        int x = kp.pt.x;
        int y = kp.pt.y;
        int sum = 0;
        uchar pixel = img.at<uchar>(y, x);
        for (const auto& pt : circonferenza) {
            int dx = x + pt.x;
            int dy = y + pt.y;
            if (dx >= 0 && dx < img.cols && dy >= 0 && dy < img.rows) {
                uchar nearest = img.at<uchar>(dy, dx);
                sum += abs(pixel - nearest);
            }
        }
        V_keypoints[i] = sum;
    }
    for (int i = 0; i < keypoints.size(); i++) {
        KeyPoint kp = keypoints[i];
        int x = kp.pt.x;
        int y = kp.pt.y;
        for (int j = 0; j < keypoints.size(); j++) {
            if (i != j) {
                KeyPoint kp2 = keypoints[j];
                int x2 = kp2.pt.x;
                int y2 = kp2.pt.y;
                if (abs(x - x2) <= dist && abs(y - y2) <= dist) {
                    if (V_keypoints[i] < V_keypoints[j]) {
                        keypoints.erase(keypoints.begin() + i);
                        i--;
                        break;
                    }
                }
            }
        }
    }
}

vector<KeyPoint> FASTCorners(const Mat& sourceImage, int threshold, bool nonMaxSuppression) {
    Mat grayImg;
    cvtColor(sourceImage, grayImg, COLOR_BGR2GRAY);
    vector<KeyPoint> keypoints;
    for (int y = 3; y < grayImg.rows - 3; y++) {
        for (int x = 3; x < grayImg.cols - 3; x++) {
            if (fastHighSpeedTest(grayImg, x, y, threshold) && fastSegmentTest(grayImg, x, y, threshold)) {
                keypoints.push_back(KeyPoint(x, y, 1));
            }
        }
    }
    if (nonMaxSuppression) {
        nonMaximumSuppression(keypoints, grayImg);
    }
    return keypoints;
}
