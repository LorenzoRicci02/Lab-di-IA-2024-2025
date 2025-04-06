#ifndef MY_ORB_H
#define MY_ORB_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

void computeORB(const Mat& image1, const Mat& image2, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, Mat& descriptors1, Mat& descriptors2, vector<DMatch>& goodMatches, int fastThreshold = 50, int maxKeypoints = 500, int briefBits = 256);

#endif
