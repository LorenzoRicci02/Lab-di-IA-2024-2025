#ifndef SHI_TOMASI_H
#define SHI_TOMASI_H

#include <opencv2/opencv.hpp>
#include <vector>

void resizeImage(const cv::Mat& src, cv::Mat& dst, int newWidth, int newHeight);
void detectShiTomasiCorners(const cv::Mat& src, cv::Mat& dst, std::vector<cv::Point2f>& corners, int maxCorners = 1000, double qualityLevel = 0.02, double minDistance = 20);
void matchCorners(const std::vector<cv::Point2f>& corners1, const std::vector<cv::Point2f>& corners2, cv::Mat& img1, cv::Mat& img2, cv::Mat& result, int& matchCount, int& unmatchedCount);

#endif
