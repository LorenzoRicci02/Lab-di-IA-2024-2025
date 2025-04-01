#include "ocv_orb.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void detectAndMatchORB(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& imgMatches) {
    // Usiamo ORB per rilevare i keypoints e estrarre i descrittori
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.5f, 8, 100);

    // Rilevo i keypoints e calcolo i descrittori per la prima immagine
    std::vector<cv::KeyPoint> keypoints1;
    cv::Mat descriptors1;
    orb->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);

    // Rilevo i keypoints e calcolo i descrittori per la seconda immagine
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat descriptors2;
    orb->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    cv::BFMatcher matcher(cv::NORM_HAMMING, false);

    // Vettore per il matching KNN
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2); 

    std::vector<cv::DMatch> all_matches;
    for (const auto& matches : knn_matches) {
        if (!matches.empty()) {
            all_matches.push_back(matches[0]);
        }
    }

    // Estraggo i punti chiave corrispondenti
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : all_matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    // Applico RANSAC per rimuovere i falsi positivi
    std::vector<uchar> inliersMask;
    if (points1.size() >= 4) {
        cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC, 5.0, inliersMask, 5000, 0.995);
    }

    // Filtro i match utilizzando la maschera degli inlier
    std::vector<cv::DMatch> inlier_matches;
    for (size_t i = 0; i < inliersMask.size(); i++) {
        if (inliersMask[i]) {
            inlier_matches.push_back(all_matches[i]); 
        }
    }

    // Disegno solo gli inlier
    cv::drawMatches(img1, keypoints1, img2, keypoints2, inlier_matches, imgMatches, cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Stampa le statistiche
    std::cout << "\nKeypoints nella prima immagine (ORB): " << keypoints1.size() << std::endl;
    std::cout << "Keypoints nella seconda immagine (ORB): " << keypoints2.size() << std::endl;
    std::cout << "Numero di match rilevati con ORB: " << inlier_matches.size() << std::endl;
}
