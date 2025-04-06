#include "ocv_orb.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

void detectAndMatchORB(const Mat& img1, const Mat& img2, Mat& imgMatches) {
    // Usiamo ORB per rilevare i keypoints e estrarre i descrittori
    Ptr<ORB> orb = ORB::create(500, 1.5f, 8, 100);

    // Rilevo i keypoints e calcolo i descrittori per la prima immagine
    vector<KeyPoint> keypoints1;
    Mat descriptors1;
    orb->detectAndCompute(img1, noArray(), keypoints1, descriptors1);

    // Rilevo i keypoints e calcolo i descrittori per la seconda immagine
    vector<KeyPoint> keypoints2;
    Mat descriptors2;
    orb->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    BFMatcher matcher(NORM_HAMMING, false);

    // Vettore per il matching KNN
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2); 

    vector<DMatch> all_matches;
    for (const auto& matches : knn_matches) {
        if (!matches.empty()) {
            all_matches.push_back(matches[0]);
        }
    }

    // Estraggo i punti chiave corrispondenti
    vector<Point2f> points1, points2;
    for (const auto& match : all_matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    // Applico RANSAC per rimuovere i falsi positivi
    vector<uchar> inliersMask;
    if (points1.size() >= 4) {
        Mat H = findHomography(points1, points2, RANSAC, 5.0, inliersMask, 5000, 0.995);
    }

    // Filtro i match utilizzando la maschera degli inlier
    vector<DMatch> inlier_matches;
    for (size_t i = 0; i < inliersMask.size(); i++) {
        if (inliersMask[i]) {
            inlier_matches.push_back(all_matches[i]); 
        }
    }

    // Disegno solo gli inlier
    drawMatches(img1, keypoints1, img2, keypoints2, inlier_matches, imgMatches, Scalar(0, 255, 0), Scalar(0, 255, 0), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Stampa le statistiche
    cout << "\nKeypoints nella prima immagine ORB (OCV): " << keypoints1.size() << endl;
    cout << "Keypoints nella seconda immagine ORB (OCV): " << keypoints2.size() << endl;
    cout << "Numero di match rilevati con ORB (OCV): " << inlier_matches.size() << endl;
}
