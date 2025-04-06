#ifndef MY_BRIEF_H
#define MY_BRIEF_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

Mat computeBRIEF(const Mat &img, const vector<KeyPoint> &keypoints, int patch_size = 32, int n_bits = 256);
vector<DMatch> matchBRIEF(const Mat &desc1, const Mat &desc2);

#endif
