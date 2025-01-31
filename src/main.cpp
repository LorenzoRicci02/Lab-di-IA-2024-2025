#include "shi-tomasi.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    string imagePath1 = "pano/selfie/s1.jpg";
    string imagePath2 = "pano/selfie/s2.jpg";

    Mat img1 = imread(imagePath1, IMREAD_COLOR);
    Mat img2 = imread(imagePath2, IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        cout << "Errore nel caricamento delle immagini!" << endl;
        return -1;
    }

    // Vettori per i corner rilevati nelle due immagini
    vector<Point2f> corners1, corners2;

    // Ridimensiona le immagini
    Mat resizedImg1, resizedImg2;
    resizeImage(img1, resizedImg1, 510, 680);
    resizeImage(img2, resizedImg2, 510, 680);

    // Applica Shi-Tomasi Corner Detection su entrambe le immagini
    Mat img_corners1, img_corners2;
    detectShiTomasiCorners(resizedImg1, img_corners1, corners1);
    detectShiTomasiCorners(resizedImg2, img_corners2, corners2);

    // Salva le immagini con i corner rilevati
    string outputCorners1 = "output/shi_tomasi_corners1.jpg";
    string outputCorners2 = "output/shi_tomasi_corners2.jpg";
    imwrite(outputCorners1, img_corners1);
    imwrite(outputCorners2, img_corners2);
    cout << "Immagini con i corner salvate in: " << outputCorners1 << " e " << outputCorners2 << endl;

    // Variabili per i match
    Mat result;
    int matchCount = 0, unmatchedCount = 0;

    // Confronta i corner tra le due immagini
    matchCorners(corners1, corners2, resizedImg1, resizedImg2, result, matchCount, unmatchedCount);

    // Salva l'immagine con i match
    string outputMatch = "output/shi_tomasi_matches.jpg";
    imwrite(outputMatch, result);

    cout << "Immagine con i match salvata in: " << outputMatch << endl;

    return 0;
}

/*
Questa per adesso è la migliore, mi risulta:
due foto uguali 100% di match ed è giusto
due foto praticamente uguali 83% di match e andrebbe aumentato di poco
due foto simili 44% di match e andrebbe aumentato
due foto completamente diverse 32% di match e andrebbe abbassato
*/