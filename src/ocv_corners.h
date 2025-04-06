#ifndef CORNER_H 
#define CORNER_H

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp> 
#include <vector>

using namespace cv;
using namespace std;

void resizeImage(const Mat& src, Mat& dst, int newWidth, int newHeight);

/* Ho scelto maxCorners = 1000 per non imporre un limite al rilevamento dei corner, la selezione effettiva dipende da qualityLevel e minDistance. 
   QualityLevel (tra 0 e 1) definisce la soglia minima di qualità per accettare un corner: valori più alti selezionano solo quelli più distintivi.  
   minDistance imposta la distanza minima tra due corner rilevati, evitando punti troppo vicini e migliorando la distribuzione dei corner nell'immagine. */
void detectShiTomasiCorners(const Mat& src, Mat& dst, vector<Point2f>& corners, int maxCorners = 1000, double qualityLevel = 0.09, double minDistance = 10);

void detectFASTCorners(const Mat& src, Mat& dst, vector<KeyPoint>& keypoints, int threshold = 45, bool nonMaxSuppression = true);

// Funzione per rilevare i corner con il metodo Harris
void detectHarrisCorners(const Mat& src, Mat& dst, double blockSize, double ksize, double k, double threshold, double minDistance, vector<KeyPoint>& keypoints);

#endif
