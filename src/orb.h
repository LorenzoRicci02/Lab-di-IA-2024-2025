#ifndef ORB_H
#define ORB_H

#include <opencv2/opencv.hpp>

void detectAndMatchORB(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& imgMatches);

#endif
