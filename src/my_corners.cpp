#include "my_corners.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// FUNZIONI SHI TOMASI

// Calcolo i gradienti con Sobel
pair<Mat, Mat> computeDerivatives(const Mat& image) {
    Mat gradX, gradY;
    Sobel(image, gradX, CV_64F, 1, 0, 3);
    Sobel(image, gradY, CV_64F, 0, 1, 3);
    return {gradX, gradY};
}

// Calcolo il più piccolo autovalore della matrice M
inline double smallestEigenvalue(double a, double b, double c) {
    double trace = a + c;
    double det = a * c - b * b;
    double delta = sqrt(trace * trace - 4 * det);
    return 0.5 * (trace - delta);
}

// Non-maximum suppression classica
Mat nonMaximumSuppression(const Mat& responseMap, int windowSize) {
    Mat filtered = Mat::zeros(responseMap.size(), CV_64F);
    int halfWindow = windowSize / 2;

    for (int row = halfWindow; row < responseMap.rows - halfWindow; row++) {
        for (int col = halfWindow; col < responseMap.cols - halfWindow; col++) {
            double current = responseMap.at<double>(row, col);
            bool isMax = true;
            for (int dy = -halfWindow; dy <= halfWindow && isMax; dy++) {
                for (int dx = -halfWindow; dx <= halfWindow; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    if (responseMap.at<double>(row + dy, col + dx) > current) {
                        isMax = false;
                        break;
                    }
                }
            }
            if (isMax) filtered.at<double>(row, col) = current;
        }
    }

    return filtered;
}


vector<KeyPoint> ShiTomasiCorners(const Mat& sourceImage, double qualityRatio, int maxPoints, int patchSize) {
    // 1. Preprocessing
    Mat grayImg;
    cvtColor(sourceImage, grayImg, COLOR_BGR2GRAY);
    grayImg.convertTo(grayImg, CV_64F);

    // 2. Calcolo Gradienti Ix e Iy
    auto [gradX, gradY] = computeDerivatives(grayImg);

    // 3. Matrice dei punteggi (autovalore min di H per ciascun pixel)
    Mat scoreMap = Mat::zeros(grayImg.size(), CV_64F);
    double peakResponse = 0;
    int halfPatch = patchSize / 2;

    for (int row = halfPatch; row < grayImg.rows - halfPatch; row++) {
        for (int col = halfPatch; col < grayImg.cols - halfPatch; col++) {
            double a = 0, b = 0, c = 0;

            // Per ciascun pixel genera la matrice H relativa
            for (int dy = -halfPatch; dy <= halfPatch; ++dy) {
                for (int dx = -halfPatch; dx <= halfPatch; ++dx) {
                    int yk = row + dy;
                    int xk = col + dx;
                    double ix = gradX.at<double>(yk, xk);
                    double iy = gradY.at<double>(yk, xk);
                    a += ix * ix;
                    b += ix * iy;
                    c += iy * iy;
                }
            }
            // Estrae il min autovalore
            double lambdaMin = smallestEigenvalue(a, b, c);
            scoreMap.at<double>(row, col) = lambdaMin;
            if (lambdaMin > peakResponse) peakResponse = lambdaMin;
        }
    }

    // 4. Soglia basata su qualità (se un pixel è massimo locale ma sotto soglia viene scartato)
    double threshold = qualityRatio * peakResponse;

    // 5. Non-maximum suppression sulla score map
    Mat suppressed = nonMaximumSuppression(scoreMap, patchSize);

    // 6. Estrazione dei keypoint
    vector<KeyPoint> keypoints;
    for (int row = 0; row < suppressed.rows; row++) {
        for (int col = 0; col < suppressed.cols; col++) {
            double val = suppressed.at<double>(row, col);
            if (val > threshold) {
                keypoints.push_back(KeyPoint(col, row, 1, -1, val));
            }
        }
    }

    // 7. Se troppi keypoint, tieni solo i migliori (opzionale nel mio caso non influisce)
    if (keypoints.size() > maxPoints) {
        sort(keypoints.begin(), keypoints.end(), [](const KeyPoint& a, const KeyPoint& b) {
            return a.response > b.response;
        });
        keypoints.resize(maxPoints);
    }

    return keypoints;
}


// FUNZIONI FAST

// Definisco una circonferenza di raggio 3 (16 pixel)
vector<Point> circonferenza = {
    {0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
    {0, 3}, {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}
};


// Test effettivo con n pixel consecutivi significativamente più scuri/più chiari del pixel target
bool fastSegmentTest(const Mat &img, int x, int y, int threshold, int n) {
    uchar pixel = img.at<uchar>(y, x);
    int countHigh = 0, countLow = 0;
    for (int i = 0; i < 16; i++) {
        int dx = x + circonferenza[i].x;
        int dy = y + circonferenza[i].y;
        if (dx < 0 || dx >= img.cols || dy < 0 || dy >= img.rows)
            continue;
        uchar nearest = img.at<uchar>(dy, dx);
        if (nearest > pixel + threshold) {
            countHigh++;
        } else if (nearest < pixel - threshold) {
            countLow++;
        }
    }
    return (countHigh >= n || countLow >= n);
}

// Test "preliminare" confronto con solo 4 pixel chiave (più rapido per questo prende il nome di highspeed)
bool fastHighSpeedTest(const Mat &img, int x, int y, int threshold) {
    uchar pixel = img.at<uchar>(y, x);
    vector<int> test_pixel = {1, 9, 5, 13};   // 4 pixel chiave p1 p5 p9 p13
    int countHigh = 0, countLow = 0;          // quanti più chiari e quanti più scuri
    for (int i = 0; i < test_pixel.size(); i++) {
        int j = test_pixel[i];
        int dx = x + (circonferenza[j].x);
        int dy = y + (circonferenza[j].y);
        if (dx < 0 || dx >= img.cols || dy < 0 || dy >= img.rows)
            continue;
        uchar nearest = img.at<uchar>(dy, dx);
        if (nearest > pixel + threshold) {
            countHigh++;
        } else if (nearest < pixel - threshold) {
            countLow++;
        }
    }
    return (countHigh >= 3 || countLow >= 3);  // test impostato su almeno 3 più chiari o 3 più scuri
}


// Non maximum suppression
void nonMaximumSuppression(vector<KeyPoint>& keypoints, int minDist) {
    // Ordina per risposta decrescente
    sort(keypoints.begin(), keypoints.end(), [](const KeyPoint& a, const KeyPoint& b) {
        return a.response > b.response;
    });

    vector<KeyPoint> finalKeypoints;
    
    for (const auto& kp : keypoints) {
        bool keep = true;
        for (const auto& selected : finalKeypoints) {
            if (norm(kp.pt - selected.pt) < minDist) {
                keep = false;
                break;
            }
        }
        if (keep) {
            finalKeypoints.push_back(kp);
        }
    }

    keypoints = finalKeypoints;
}



vector<KeyPoint> FASTCorners(const Mat& sourceImage, int threshold, bool nonMaxSuppression) {
    Mat grayImg;
    cvtColor(sourceImage, grayImg, COLOR_BGR2GRAY);
    vector<KeyPoint> keypoints;
    for (int y = 3; y < grayImg.rows - 3; y++) {
        for (int x = 3; x < grayImg.cols - 3; x++) {
            if (fastHighSpeedTest(grayImg, x, y, threshold) && fastSegmentTest(grayImg, x, y, threshold)) { // Solo se passa il primo test preliminare si passa al secondo
                keypoints.push_back(KeyPoint(x, y, 1));
            }
        }
    }
    if (nonMaxSuppression) {
        nonMaximumSuppression(keypoints, 5);
    }
    return keypoints;
}

// FUNZIONI HARRIS (Vedi esercitazione 4 HarrisImage.cpp fatta da me)

// Calcolo i gradienti con Sobel
pair<Mat, Mat> gradient(const Mat& img) {
    Mat Ix, Iy;

    // Calcola il gradiente usando Sobel
    Sobel(img, Ix, CV_32F, 1, 0, 3);  // Gradiente in X
    Sobel(img, Iy, CV_32F, 0, 1, 3);  // Gradiente in Y

    return {Ix, Iy};
}

// Calcolo Ix^2, Iy^2, IxIy
tuple<Mat, Mat, Mat> structureTensor(const Mat& Ix, const Mat& Iy, int windowSize) {
    int halfWindow = windowSize / 2;
    Mat SxSx = Mat::zeros(Ix.size(), CV_32F);
    Mat SySy = Mat::zeros(Ix.size(), CV_32F);
    Mat SxSy = Mat::zeros(Ix.size(), CV_32F);

    for (int y = halfWindow; y < Ix.rows - halfWindow; y++) {
        for (int x = halfWindow; x < Ix.cols - halfWindow; x++) {
            float sumIx2 = 0.0f;
            float sumIy2 = 0.0f;
            float sumIxIy = 0.0f;

            for (int dy = -halfWindow; dy <= halfWindow; dy++) {
                for (int dx = -halfWindow; dx <= halfWindow; dx++) {
                    float ix = Ix.at<float>(y + dy, x + dx);
                    float iy = Iy.at<float>(y + dy, x + dx);
                    sumIx2 += ix * ix;
                    sumIy2 += iy * iy;
                    sumIxIy += ix * iy;
                }
            }

            SxSx.at<float>(y, x) = sumIx2;
            SySy.at<float>(y, x) = sumIy2;
            SxSy.at<float>(y, x) = sumIxIy;
        }
    }

    return {SxSx, SySy, SxSy};
}


// Calcolo la risposta di Harris (Determinante e Traccia)
Mat harrisResponse(const Mat& SxSx, const Mat& SySy, const Mat& SxSy) {
    Mat corners = Mat::zeros(SxSx.size(), CV_32F);
    
    for (int y = 0; y < SxSx.rows; y++) {
        for (int x = 0; x < SxSx.cols; x++) {
            float a = SxSx.at<float>(y, x);
            float b = SxSy.at<float>(y, x);
            float c = SySy.at<float>(y, x);
            
            float det = a * c - b * b;
            float trace = a + c;

            if (trace != 0) {
                corners.at<float>(y, x) = det / trace; 
            } else {
                corners.at<float>(y, x) = 0;
            }
        }
    }

    // Normalizzo i valori della risposta
    double minVal, maxVal;
    minMaxLoc(corners, &minVal, &maxVal);
    for (int y = 0; y < corners.rows; ++y) {
        for (int x = 0; x < corners.cols; ++x) {
            float& val = corners.at<float>(y, x);
            val = 255.0f * (val - (float)minVal) / ((float)maxVal - (float)minVal + 1e-10f);
        }
    }
    
    return corners;
}


// Non maximum suppression personalizzata per Harris (quella di shitomasi non riesco ad usarla)
Mat NMS_Harris(const Mat& corners, float threshold) {
    Mat cornersSuppressed = Mat::zeros(corners.size(), CV_32F);

    for (int y = 1; y < corners.rows - 1; y++) {
        for (int x = 1; x < corners.cols - 1; x++) {
            float pixel = corners.at<float>(y, x);
            bool localMax = true;

            // Controlla se è il massimo locale in una finestra 3x3
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (pixel < corners.at<float>(y + dy, x + dx)) {
                        localMax = false;
                    }
                }
                if (!localMax) break;
            }

            // Se è massimo locale e supera la soglia
            if (localMax && pixel > threshold) {
                cornersSuppressed.at<float>(y, x) = pixel;
            }
        }
    }

    return cornersSuppressed;
}

// Estraggo i keypoint 
vector<KeyPoint> extractKeypoints(const Mat& corners) {
    vector<KeyPoint> keypoints;
    
    for (int y = 0; y < corners.rows; y++) {
        for (int x = 0; x < corners.cols; x++) {
            if (corners.at<float>(y, x) > 0) {
                KeyPoint keypoint(Point2f(x, y), 1);
                keypoints.push_back(keypoint);
            }
        }
    }
    
    return keypoints;
}

vector<KeyPoint> HarrisCorners(const Mat& imgRGB, int window, float threshold) {
    // Pre-processing dell'immagine
    Mat img;
    cvtColor(imgRGB, img, COLOR_BGR2GRAY);

    // Step 1. Calcola i gradienti spaziali
    auto [Ix, Iy] = gradient(img);

    // Step 2. Calcola Ix^2, Iy^2, IxIy
    auto [SxSx, SySy, SxSy] = structureTensor(Ix, Iy, window);

    // Step 3. Calcola la risposta di Harris
    Mat cornersNonSuppressed = harrisResponse(SxSx, SySy, SxSy);

    // Step 4. Calcolo la treshold dinamnica basandomi sulla media e sulla risposta massima
    double maxResponse = 0;
    double meanResponse = 0;
    minMaxLoc(cornersNonSuppressed, nullptr, &maxResponse);
    meanResponse = mean(cornersNonSuppressed).val[0];
    
    float dynamicThreshold = threshold * (float)(meanResponse + maxResponse) / 2;

    // Step 5. Non maximum suppression (NMS) con soglia dinamica
    Mat corners = NMS_Harris(cornersNonSuppressed, dynamicThreshold);

    // Step 6. Estraggo i keypoints
    vector<KeyPoint> keypoints = extractKeypoints(corners);

    // Step 7. Rimozione dei keypoints troppo vicini (utile nella mia implementazione ma opzionale)
    vector<KeyPoint> filteredKeypoints;
    const int minDistance = 8; 
    for (const auto& kp : keypoints) {
        bool tooClose = false;
        for (const auto& filteredKp : filteredKeypoints) {
            if (norm(kp.pt - filteredKp.pt) < minDistance) {
                tooClose = true;
                break;
            }
        }
        if (!tooClose) {
            filteredKeypoints.push_back(kp);
        }
    }

    return filteredKeypoints;
}



