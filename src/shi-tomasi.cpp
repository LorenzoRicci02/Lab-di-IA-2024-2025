#include "shi-tomasi.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;


// Funzione per ridimensionare l'immagine sfruttando l'interpolazione bilineare
void resizeImage(const Mat& src, Mat& dst, int newWidth, int newHeight) {
    Size newSize(newWidth, newHeight);
    resize(src, dst, newSize, 0, 0, INTER_LINEAR);
}

void detectShiTomasiCorners(const Mat& src, Mat& dst, vector<Point2f>& corners, int maxCorners, double qualityLevel, double minDistance) {
    // Converto l'immagine in scala di grigi
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    // Trovo i corner con Shi-Tomasi
    goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance, Mat(), 3, false, 0.04);

    // Disegnamo i corner appena rilevati
    dst = src.clone();
    for (size_t i = 0; i < corners.size(); i++) {
        circle(dst, corners[i], 3, Scalar(0, 255, 0), FILLED);
    }
}

void matchCorners(const vector<Point2f>& corners1, const vector<Point2f>& corners2, Mat& img1, Mat& img2, Mat& result, int& matchCount, int& unmatchedCount) {
    
    // Crea un imamgine contenente a sx l'immagine 1 e a dx l'immagine 2
    result = Mat::zeros(max(img1.rows, img2.rows), img1.cols + img2.cols, img1.type());
    img1.copyTo(result(Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(result(Rect(img1.cols, 0, img2.cols, img2.rows)));

    // Variabili ausiliarie per il matching
    matchCount = 0;
    unmatchedCount = 0;

    // Vettori per le corrispondenze tra corner
    vector<Point2f> matchedCorners1, matchedCorners2;

    // Vettori per le non corrispondenze tra corner
    vector<Point2f> unmatchedCorners1, unmatchedCorners2;

    // Trovo la distanza media tra i corner in corners1 per calibrare la soglia dinamica
    double avgDist = 0.0;
    for (size_t i = 0; i < corners1.size(); i++) {
        for (size_t j = i + 1; j < corners1.size(); j++) {
            avgDist += norm(corners1[i] - corners1[j]);
        }
    }
    avgDist /= (corners1.size() * (corners1.size() - 1)) / 2;

    // Definisco una soglia dinamica basata sulla distanza media
    double dynamicThreshold = avgDist * 1.2; 

    // Confrontiamo i corner tra le due immagini (salvati nei vettori corners1 e corners2)
    for (size_t i = 0; i < corners1.size(); i++) {
        double minDist = DBL_MAX;      
        double secondMinDist = DBL_MAX; 
        size_t bestMatch = -1;             
        size_t secondBestMatch = -1;

        // Trova il miglior match e il secondo miglior match tra corners1[i] e corners2[j]
        for (size_t j = 0; j < corners2.size(); j++) {
            double dist = norm(corners1[i] - corners2[j]); // calcolo la distanza euclidea tra i 2 corner

            // Trova il miglior match
            if (dist < minDist) {
                secondMinDist = minDist;
                secondBestMatch = bestMatch;
                minDist = dist;
                bestMatch = j;
            }
            // Trova il secondo miglior match
            else if (dist < secondMinDist) {
                secondMinDist = dist;
                secondBestMatch = j;
            }
        }

        // Applica il Ratio Test di Lowe con la soglia dinamica
        if (minDist < 0.7 * secondMinDist && minDist < dynamicThreshold) {
            matchedCorners1.push_back(corners1[i]);
            matchedCorners2.push_back(corners2[bestMatch]);

            // Disegno i match sull'immagine
            line(result, corners1[i], corners2[bestMatch] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 1);
            circle(result, corners1[i], 3, Scalar(0, 255, 0), FILLED);
            circle(result, corners2[bestMatch] + Point2f(img1.cols, 0), 3, Scalar(0, 255, 0), FILLED);

            matchCount++;
        } else { 
            // Se il match fallisce disegno gli unmatched points
            unmatchedCorners1.push_back(corners1[i]);
            unmatchedCorners2.push_back(corners2[bestMatch]);
        }
    }

    unmatchedCount = corners1.size() - matchCount;

    // Collego i corner non corrispondenti con una linea rossa
    for (size_t i = 0; i < unmatchedCorners1.size(); i++) {
        line(result, unmatchedCorners1[i], unmatchedCorners2[i] + Point2f(img1.cols, 0), Scalar(0, 0, 255), 1);
        circle(result, unmatchedCorners1[i], 3, Scalar(0, 0, 255), FILLED);
        circle(result, unmatchedCorners2[i] + Point2f(img1.cols, 0), 3, Scalar(0, 0, 255), FILLED);
    }

    // Calcolo della percentuale di matching tra le due foto
    double matchPercentage = round((static_cast<double>(matchCount) / max(corners1.size(), corners2.size())) * 1000) / 10.0;

    cout << "\nNumero di match trovati (inliers in verde): " << matchCount << endl;
    cout << "Numero di match errati (outliers in rosso): " << unmatchedCorners1.size() << endl;
    cout << "Percentuale di matching geometrico: " << matchPercentage << "%" << endl;
}

