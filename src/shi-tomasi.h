#ifndef SHI_TOMASI_H 
#define SHI_TOMASI_H

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp> 
#include <vector>

void resizeImage(const cv::Mat& src, cv::Mat& dst, int newWidth, int newHeight);

/* Ho scelto maxCorners = 1000 per non imporre un limite al rilevamento dei corner, la selezione effettiva dipende da qualityLevel e minDistance. 
   QualityLevel (tra 0 e 1) definisce la soglia minima di qualità per accettare un corner: valori più alti selezionano solo quelli più distintivi.  
   minDistance imposta la distanza minima tra due corner rilevati, evitando punti troppo vicini e migliorando la distribuzione dei corner nell'immagine. */
void detectShiTomasiCorners(const cv::Mat& src, cv::Mat& dst, std::vector<cv::Point2f>& corners, int maxCorners = 1000, double qualityLevel = 0.02, double minDistance = 20);

void matchCorners(const std::vector<cv::Point2f>& corners1, const std::vector<cv::Point2f>& corners2, cv::Mat& img1, cv::Mat& img2, cv::Mat& result, int& matchCount, int& unmatchedCount);

// Funzione per disegnare i match tra le due immagini
void drawDescriptorMatches(const cv::Mat& img1, const cv::Mat& img2, const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2, const std::vector<cv::DMatch>& matches, cv::Mat& result);

#endif
