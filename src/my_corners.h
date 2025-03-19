#ifndef MY_CORNERS_H
#define MY_CORNERS_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

pair<Mat, Mat> computeDerivatives(const Mat& image);

Matx22d buildStructureMatrix(int posX, int posY, const Mat& gradX, const Mat& gradY, int kernelSize);

vector<double> extractEigenvalues(const Matx22d& structMatrix);

Mat nonMaximumSuppression(const Mat& responseMap, int windowSize);

vector<KeyPoint> ShiTomasiCorners(const Mat& sourceImage, double qualityRatio, int maxPoints, int patchSize);

#endif
