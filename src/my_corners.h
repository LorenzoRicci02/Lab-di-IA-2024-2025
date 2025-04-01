#ifndef MY_CORNERS_H
#define MY_CORNERS_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// Shi-Tomasi
pair<Mat, Mat> computeDerivatives(const Mat& image);
Matx22d buildStructureMatrix(int posX, int posY, const Mat& gradX, const Mat& gradY, int kernelSize);
vector<double> extractEigenvalues(const Matx22d& structMatrix);
Mat nonMaximumSuppression(const Mat& responseMap, int windowSize);
vector<KeyPoint> ShiTomasiCorners(const Mat& sourceImage, double qualityRatio, int maxPoints, int patchSize);

// FAST
bool fastSegmentTest(const Mat &img, int x, int y, int threshold = 50, int n = 8);
bool fastHighSpeedTest(const Mat &img, int x, int y, int threshold);
void nonMaximumSuppression(vector<KeyPoint> &keypoints, const Mat &img, int dist = 3);
vector<KeyPoint> FASTCorners(const Mat& sourceImage, int threshold, bool nonMaxSuppression);

#endif 
