#include "my_corners.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

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
