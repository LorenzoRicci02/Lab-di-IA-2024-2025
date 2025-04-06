#ifndef ORB_H
#define ORB_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void detectAndMatchORB(const Mat& img1, const Mat& img2, Mat& imgMatches);

#endif
