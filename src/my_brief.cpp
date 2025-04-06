#include "my_brief.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include <bitset>

using namespace cv;
using namespace std;

// Pattern globale
vector<Point> briefPattern;

// Genera un pattern di coppie di punti, usato da BRIEF per confronti di intensità.
void generateBriefPattern(int n_pairs, int patch_size) {
    briefPattern.clear();
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(-patch_size / 2, patch_size / 2);
    for (int i = 0; i < 2 * n_pairs; i++) {
        briefPattern.emplace_back(dist(gen), dist(gen));
    }
}

// Ruoto un punto attorno all'origine di un certo angolo (in radianti)
Point rotatePoint(const Point& p, float angle) {
    float cosA = cos(angle);
    float sinA = sin(angle);
    return Point(cvRound(p.x * cosA - p.y * sinA), cvRound(p.x * sinA + p.y * cosA));
}

// Estraggo i descrittori BRIEF da una lista di keypoint 
Mat computeBRIEF(const Mat &img, const vector<KeyPoint> &keypoints, int patch_size, int n_bits) {
    if (briefPattern.size() != 2 * n_bits)
        generateBriefPattern(n_bits, patch_size);

    Mat gray;
    if (img.channels() == 3)
        cvtColor(img, gray, COLOR_BGR2GRAY);
    else
        gray = img.clone();

    // Matrice dei descrittori (ogni riga corrisponde a un keypoint)
    Mat descriptors = Mat::zeros((int)keypoints.size(), n_bits / 8, CV_8U);

    for (int i = 0; i < keypoints.size(); ++i) {
        const KeyPoint& kp = keypoints[i];
        float angle = kp.angle * static_cast<float>(CV_PI / 180.0); // in radianti
        Point center(cvRound(kp.pt.x), cvRound(kp.pt.y));

        bitset<256> desc;
        
        // Per ogni keypoint
        for (int j = 0; j < n_bits; ++j) {
            Point p1 = rotatePoint(briefPattern[2 * j], angle) + center;
            Point p2 = rotatePoint(briefPattern[2 * j + 1], angle) + center;

            if (p1.inside(Rect(0, 0, gray.cols, gray.rows)) &&
                p2.inside(Rect(0, 0, gray.cols, gray.rows))) {
                desc[j] = gray.at<uchar>(p1) < gray.at<uchar>(p2);
            }
        }

        // Converto i bit del descrittore in byte e li inserisce nella matrice
        for (int j = 0; j < n_bits / 8; ++j) {
            uchar byte = 0;
            for (int b = 0; b < 8; ++b) {
                byte |= desc[j * 8 + b] << b;
            }
            descriptors.at<uchar>(i, j) = byte;
        }
    }

    return descriptors;
}

// Calcola la distanza di Hamming tra due byte (numero di bit diversi)
int hammingDistanceByte(uchar a, uchar b) {
    uchar val = a ^ b;
    int count = 0;
    while (val) {
        count += val & 1;
        val >>= 1;
    }
    return count;
}

// Matcho i descrittori BRIEF utilizzando la distanza di Hamming calcolata sopra
vector<DMatch> matchBRIEF(const Mat &desc1, const Mat &desc2) {
    vector<DMatch> matches;

    // Confronto i descrittori del primo set
    for (int i = 0; i < desc1.rows; ++i) {
        int bestDist = INT_MAX;
        int bestIdx = -1;

        // Con ogni descrittore del secondo set byte per byte
        for (int j = 0; j < desc2.rows; ++j) {
            int dist = 0;
            for (int k = 0; k < desc1.cols; ++k) {
                uchar byte1 = desc1.at<uchar>(i, k);
                uchar byte2 = desc2.at<uchar>(j, k);
                dist += hammingDistanceByte(byte1, byte2);
            }
            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = j;
            }
        }

        // Salvo il miglior match aggiungendolo alla lista
        if (bestIdx >= 0) {
            matches.emplace_back(i, bestIdx, (float)bestDist);
        }
    }

    return matches;
}

