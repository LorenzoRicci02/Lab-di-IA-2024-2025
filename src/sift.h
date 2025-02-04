#ifndef SIFT_H
#define SIFT_H

#include <opencv2/opencv.hpp>

void detectAndMatchSIFT(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& imgMatches);

#endif // SIFT_H
