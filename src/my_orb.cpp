#include "my_orb.h"
#include "my_corners.h"
#include "my_brief.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

// Calcola l'orientamento di un keypoint per garantire l'invarianza alla rotazione (BRIEF di per se non è rotation invariant!)
float computeOrientation(const Mat& image, const KeyPoint& kp, int patchSize = 31) {
    int x0 = round(kp.pt.x);
    int y0 = round(kp.pt.y);
    int half = patchSize / 2;

    Moments m;
    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {
            int x = x0 + dx;
            int y = y0 + dy;
            if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
                uchar val = image.at<uchar>(y, x);
                m.m00 += val;
                m.m10 += x * val;
                m.m01 += y * val;
            }
        }
    }

    if (m.m00 == 0) return 0.0f;
    float cx = m.m10 / m.m00;
    float cy = m.m01 / m.m00;

    return atan2(cy - y0, cx - x0) * 180.0f / CV_PI;
}

// Filtro RANSAC sui match per eliminare gli outliers
vector<DMatch> filterMatchesRANSAC(const vector<DMatch>& matches, const vector<KeyPoint>& kp1, const vector<KeyPoint>& kp2, double ransacThresh = 3.0) {

    vector<Point2f> pts1, pts2;
    for (const DMatch& m : matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }

    vector<char> inliersMask;
    findHomography(pts1, pts2, RANSAC, ransacThresh, inliersMask);

    vector<DMatch> inlierMatches;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (inliersMask[i]) {
            inlierMatches.push_back(matches[i]);
        }
    }

    return inlierMatches;
}

void computeORB(const Mat& image1, const Mat& image2, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, Mat& descriptors1, Mat& descriptors2, vector<DMatch>& goodMatches, int fastThreshold, int maxKeypoints, int briefBits) {
    
    // Pre-processing dell'immagine
    Mat gray1, gray2;
    cvtColor(image1, gray1, COLOR_BGR2GRAY);
    cvtColor(image2, gray2, COLOR_BGR2GRAY);

    // Detecto i corner (keypoints) tramite FAST reimplementato da me
    keypoints1 = FASTCorners(image1, fastThreshold, true);
    keypoints2 = FASTCorners(image2, fastThreshold, true);

    // Limite sul numero di keypoints (superfluo nel mio caso)
    if (keypoints1.size() > maxKeypoints) keypoints1.resize(maxKeypoints);
    if (keypoints2.size() > maxKeypoints) keypoints2.resize(maxKeypoints);

    // Assegna orientamento a ciascun keypoint (img 1 e img2)
    for (auto& kp : keypoints1)
        kp.angle = computeOrientation(gray1, kp);
    for (auto& kp : keypoints2)
        kp.angle = computeOrientation(gray2, kp);

    // Utilizzo la mia implementazione di BRIEF per estrarre i descrittori dal vettore di keypoints
    descriptors1 = computeBRIEF(image1, keypoints1, 32, briefBits);
    descriptors2 = computeBRIEF(image2, keypoints2, 32, briefBits);

    // Effettuo il matching sempre con la mia funzione reimplementata
    vector<DMatch> matches = matchBRIEF(descriptors1, descriptors2);
    goodMatches = filterMatchesRANSAC(matches, keypoints1, keypoints2);

    cout << "Keypoints nella prima immagine ORB (MY): " << keypoints1.size() << endl;
    cout << "Keypoints nella seconda immagine ORB (MY): " << keypoints2.size() << endl;
    cout << "Numero di match rilevati con ORB (MY): " << goodMatches.size() << endl;
}
