#include "shi-tomasi.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void resizeImage(const Mat& src, Mat& dst, int newWidth, int newHeight) {
    Size newSize(newWidth, newHeight);
    resize(src, dst, newSize, 0, 0, INTER_LINEAR);
}

void detectShiTomasiCorners(const Mat& src, Mat& dst, vector<Point2f>& corners, int maxCorners, double qualityLevel, double minDistance) {
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    // Trova i corner con Shi-Tomasi
    goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance, Mat(), 3, false, 0.04);

    dst = src.clone();
    for (size_t i = 0; i < corners.size(); i++) {
        circle(dst, corners[i], 3, Scalar(0, 255, 0), FILLED);
    }

    cout << "Numero di corner rilevati: " << corners.size() << endl;
}

void matchCorners(const vector<Point2f>& corners1, const vector<Point2f>& corners2, Mat& img1, Mat& img2, Mat& result, int& matchCount, int& unmatchedCount) {
    result = Mat::zeros(max(img1.rows, img2.rows), img1.cols + img2.cols, img1.type());
    img1.copyTo(result(Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(result(Rect(img1.cols, 0, img2.cols, img2.rows)));

    matchCount = 0;
    unmatchedCount = 0;

    // Crea i vettori per le corrispondenze tra i corner
    vector<Point2f> matchedCorners1, matchedCorners2;

    // Trova la distanza media tra i corner in corners1 per calibrare una soglia dinamica
    double avgDist = 0.0;
    for (size_t i = 0; i < corners1.size(); i++) {
        for (size_t j = i + 1; j < corners1.size(); j++) {
            avgDist += norm(corners1[i] - corners1[j]);
        }
    }
    avgDist /= (corners1.size() * (corners1.size() - 1)) / 2;

    // Definiamo una soglia dinamica basata sulla distanza media
    double dynamicThreshold = avgDist * 1.2;  // Adattiamo la soglia dinamica

    // Confrontiamo i corner tra le due immagini
    for (size_t i = 0; i < corners1.size(); i++) {
        double minDist = DBL_MAX;
        double secondMinDist = DBL_MAX;
        size_t bestMatch = -1;
        size_t secondBestMatch = -1;

        // Trova il miglior match e il secondo miglior match nell'altra immagine
        for (size_t j = 0; j < corners2.size(); j++) {
            double dist = norm(corners1[i] - corners2[j]);

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

            line(result, corners1[i], corners2[bestMatch] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 1);
            circle(result, corners1[i], 3, Scalar(0, 255, 0), FILLED);
            circle(result, corners2[bestMatch] + Point2f(img1.cols, 0), 3, Scalar(0, 255, 0), FILLED);

            matchCount++;
        }
    }

    unmatchedCount = corners1.size() - matchCount;
    cout << "Numero di match trovati: " << matchCount << endl;
    cout << "Numero di corner non matchati in immagine 1: " << unmatchedCount << endl;
    cout << "Numero di corner non matchati in immagine 2: " << corners2.size() - matchCount << endl;
}
