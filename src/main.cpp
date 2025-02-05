#include "shi-tomasi.h"
#include "ml.h"
#include "sift.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Path per le immagini da leggere 
    string imagePath1 = "pano/selfie/s3.jpg";
    string imagePath2 = "pano/selfie/s3_flip.jpg";

    Mat img1 = imread(imagePath1, IMREAD_COLOR);
    Mat img2 = imread(imagePath2, IMREAD_COLOR);

    // Check sulle immagini non vuote
    if (img1.empty() || img2.empty()) {
        cout << "Errore nel caricamento delle immagini!" << endl;
        return -1;
    }

    // Ridimensiona le immagini per garantire un confronto preciso
    Mat resizedImg1, resizedImg2;
    resizeImage(img1, resizedImg1, 510, 680);
    resizeImage(img2, resizedImg2, 510, 680);


    //////////////  PARTE MACHINE LEARNING CON MODELLI PRE-ADDESTRATI ////////////////

    // Path per i file Haar
    string face_classifier_path = "opencv/data/haarcascades/haarcascade_frontalface_alt2.xml";
    string eyes_classifier_path = "opencv/data/haarcascades/haarcascade_eye.xml";
    string smile_classifier_path = "opencv/data/haarcascades/haarcascade_smile.xml";

    // Carica i classificatori
    CascadeClassifier faceCascade, eyesCascade, smileCascade;
    if (!loadClassifiers(face_classifier_path, eyes_classifier_path, smile_classifier_path, faceCascade, eyesCascade, smileCascade)) {
        return -1;
    }

    Mat imgFace1 = resizedImg1.clone();
    Mat imgFace2 = resizedImg2.clone();

    vector<Rect> faces1, faces2;
    detectFaces(imgFace1, faces1, faceCascade);
    drawFaceFeatures(imgFace1, faces1, eyesCascade, smileCascade);

    detectFaces(imgFace2, faces2, faceCascade);
    drawFaceFeatures(imgFace2, faces2, eyesCascade, smileCascade);

    imwrite("output/face_features1.jpg", imgFace1);
    imwrite("output/face_features2.jpg", imgFace2);

    cout << "Rilevamento completato, risultati del rilevamento salvati.\n";

    ///////////////////////////  PARTE COMPUTER VISION ///////////////////////////

    Mat imgCorners1 = resizedImg1.clone();
    Mat imgCorners2 = resizedImg2.clone();

    vector<Point2f> corners1, corners2;
    detectShiTomasiCorners(imgCorners1, imgCorners1, corners1, 100, 0.01, 10);
    detectShiTomasiCorners(imgCorners2, imgCorners2, corners2, 100, 0.01, 10);

    imwrite("output/shi_tomasi_corners1.jpg", imgCorners1);
    imwrite("output/shi_tomasi_corners2.jpg", imgCorners2);

    cout << "\nImmagini con i corner Shi-Tomasi salvate.";

    Mat matchShiTomasi;
    int matchCount = 0, unmatchedCount = 0;
    
    Mat matchImg1 = resizedImg1.clone();
    Mat matchImg2 = resizedImg2.clone();
    
    matchCorners(corners1, corners2, matchImg1, matchImg2, matchShiTomasi, matchCount, unmatchedCount);

    imwrite("output/shi_tomasi_matches.jpg", matchShiTomasi);
    cout << "Immagine con i match Shi-Tomasi salvata.\n";

    Mat siftImg1 = resizedImg1.clone();
    Mat siftImg2 = resizedImg2.clone();

    Mat siftMatches;
    detectAndMatchSIFT(siftImg1, siftImg2, siftMatches);

    imwrite("output/sift_matches.jpg", siftMatches);
    cout << "Immagine con i match SIFT salvata.\n";

    return 0;
}
