#ifndef CORNER_H 
#define CORNER_H

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp> 
#include <vector>

void resizeImage(const cv::Mat& src, cv::Mat& dst, int newWidth, int newHeight);

/* Ho scelto maxCorners = 1000 per non imporre un limite al rilevamento dei corner, la selezione effettiva dipende da qualityLevel e minDistance. 
   QualityLevel (tra 0 e 1) definisce la soglia minima di qualità per accettare un corner: valori più alti selezionano solo quelli più distintivi.  
   minDistance imposta la distanza minima tra due corner rilevati, evitando punti troppo vicini e migliorando la distribuzione dei corner nell'immagine. */
void detectShiTomasiCorners(const cv::Mat& src, cv::Mat& dst, std::vector<cv::Point2f>& corners, int maxCorners = 1000, double qualityLevel = 0.02, double minDistance = 20);

void detectFASTCorners(const cv::Mat& src, cv::Mat& dst, std::vector<cv::KeyPoint>& keypoints, int threshold = 45, bool nonMaxSuppression = true);

#endif
