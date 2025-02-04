#include "sift.h"  
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

void detectAndMatchSIFT(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& imgMatches) {
    // Inizializza il detector SIFT
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // Rileva i keypoints e calcola i descrittori per la prima immagine
    std::vector<cv::KeyPoint> keypoints1;
    cv::Mat descriptors1;
    sift->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);

    // Rileva i keypoints e calcola i descrittori per la seconda immagine
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat descriptors2;
    sift->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    // Usa un matcher KNN (FLANN o BFMatcher con K=2)
    cv::FlannBasedMatcher matcher;  // FLANN matcher, più veloce per immagini più grandi

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);  // 2 vicini più prossimi

    // Applica il Lowe's ratio test per selezionare solo le migliori corrispondenze
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < 0.79 * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    // Aggiungi il filtro per visualizzare solo le corrispondenze di buona qualità
    double max_dist = 0;
    double min_dist = 100;
    
    // Trova il massimo e minimo della distanza di matching
    for (int i = 0; i < good_matches.size(); i++) {
        double dist = good_matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    // Disegna le corrispondenze sulle due immagini
    cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, imgMatches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Stampa i contatori
    std::cout << "Keypoints nella prima immagine: " << keypoints1.size() << std::endl;
    std::cout << "Keypoints nella seconda immagine: " << keypoints2.size() << std::endl;
    std::cout << "Numero di match trovati: " << good_matches.size() << std::endl;
}
