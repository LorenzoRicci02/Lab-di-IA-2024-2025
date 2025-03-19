#include "ocv_corners.h"
#include "my_corners.h"
#include "ml.h"
#include "orb.h"
#include "sift.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono> 

using namespace cv;
using namespace std;
using namespace std::chrono;

int main() {
    // Percorsi per le immagini da utilizzare nella parte di Machine Learning
    string imagePathML1 = "pano/selfie/s1.jpg";
    string imagePathML2 = "pano/selfie/group2.jpg";

    // Percorsi per le immagini da utilizzare nella parte di Computer Vision
    string imagePathCV1 = "pano/celeb/lm.jpg";
    string imagePathCV2 = "pano/flip/lm_totalflip.jpg";

    // Carico le immagini per il machine learning
    Mat imgML1 = imread(imagePathML1, IMREAD_COLOR);
    Mat imgML2 = imread(imagePathML2, IMREAD_COLOR);

    // Controllo se le immagini non sono vuote
    if (imgML1.empty() || imgML2.empty()) {
        cout << "Errore nel caricamento delle immagini per Machine Learning!" << endl;
        return -1;
    }

    Mat resizedImgML1;
    resizeImage(imgML1, resizedImgML1, 510, 680);

    // Parte Machine Learning
    cout << "- Parte Machine Learning:" << endl;
    string face_classifier_path = "opencv/data/haarcascades/haarcascade_frontalface_alt2.xml";
    string eyes_classifier_path = "opencv/data/haarcascades/haarcascade_eye.xml";
    string smile_classifier_path = "opencv/data/haarcascades/haarcascade_smile.xml";

    CascadeClassifier faceCascade, eyesCascade, smileCascade;
    if (!loadClassifiers(face_classifier_path, eyes_classifier_path, smile_classifier_path, faceCascade, eyesCascade, smileCascade)) {
        return -1;
    }

    Mat imgFace1 = resizedImgML1.clone();
    Mat imgFace2 = imgML2.clone(); // La seconda non la ridimensiono (foto di gruppo)

    vector<Rect> faces1, faces2;
    detectFaces(imgFace1, faces1, faceCascade);
    drawFaceFeatures(imgFace1, faces1, eyesCascade, smileCascade);

    detectFaces(imgFace2, faces2, faceCascade);
    drawFaceFeatures(imgFace2, faces2, eyesCascade, smileCascade);

    imwrite("output/face_features1.jpg", imgFace1);
    imwrite("output/face_features2.jpg", imgFace2);

    cout << "Rilevamento completato per Machine Learning." << endl;

    // Carica le immagini per la parte di Computer Vision
    Mat imgCV1 = imread(imagePathCV1, IMREAD_COLOR);
    Mat imgCV2 = imread(imagePathCV2, IMREAD_COLOR);

    // Controllo se le immagini non sono vuote
    if (imgCV1.empty() || imgCV2.empty()) {
        cout << "Errore nel caricamento delle immagini per Computer Vision!" << endl;
        return -1;
    }

    // Parte Corner Detection
    cout << "\n- Parte Corner Detection:" << endl;

    // Misura il tempo per la funzione detectShiTomasiCorners
    auto start = high_resolution_clock::now();
    Mat imgCorners1_1 = imgCV1.clone();
    Mat imgCorners2_1 = imgCV2.clone();

    vector<Point2f> corners1_1, corners2_1;
    detectShiTomasiCorners(imgCorners1_1, imgCorners1_1, corners1_1);
    detectShiTomasiCorners(imgCorners2_1, imgCorners2_1, corners2_1);

    imwrite("output/ocv_shi1.jpg", imgCorners1_1);
    imwrite("output/ocv_shi2.jpg", imgCorners2_1);
    auto stop = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(stop - start);
    cout << "\nTempo di esecuzione per Shi-Tomasi con OpenCV: " << duration1.count() << " ms" << endl;

    cout << "Immagini con i corner Shi-Tomasi salvate." << endl;

    // Misura il tempo per la funzione ShiTomasiCorners
    start = high_resolution_clock::now();
    Mat imgCorners1_2 = imgCV1.clone();
    Mat imgCorners2_2 = imgCV2.clone();

    vector<KeyPoint> corners1_2, corners2_2;

    // Eseguiamo la rilevazione dei corner Shi-Tomasi per entrambe le immagini
    corners1_2 = ShiTomasiCorners(imgCorners1_2, 0.05, 1000, 7);
    corners2_2 = ShiTomasiCorners(imgCorners2_2, 0.05, 1000, 7);

    // Disegniamo i corner appena rilevati
    Mat dst1 = imgCorners1_2.clone();
    Mat dst2 = imgCorners2_2.clone();

    for (size_t i = 0; i < corners1_2.size(); i++) {
        circle(dst1, corners1_2[i].pt, 3, Scalar(0, 255, 0), FILLED);
    }

    for (size_t i = 0; i < corners2_2.size(); i++) {
        circle(dst2, corners2_2[i].pt, 3, Scalar(0, 255, 0), FILLED);
    }

    // Salviamo le immagini con i corner disegnati
    imwrite("output/my_shi1.jpg", dst1);
    imwrite("output/my_shi2.jpg", dst2);
    stop = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(stop - start);
    cout << "Tempo di esecuzione per Shi-Tomasi reimplementato da me: " << duration2.count() << " ms" << endl;

    cout << "Immagini con i corner Shi-Tomasi salvate." << endl;

    Mat imgFAST1 = imgCV1.clone();
    Mat imgFAST2 = imgCV2.clone();
    vector<KeyPoint> fastCorners1, fastCorners2;

    // Eseguo FAST su entrambe le immagini
    detectFASTCorners(imgFAST1, imgFAST1, fastCorners1);  
    detectFASTCorners(imgFAST2, imgFAST2, fastCorners2);

    // Salvo le immagini con i corner FAST
    imwrite("output/corner_fast1.jpg", imgFAST1);
    imwrite("output/corner_fast2.jpg", imgFAST2);

    cout << "Immagini con i corner FAST salvate." << endl;

    // Parte Descriptor Matching
    cout << "\n- Parte Descriptor Matching (ORB, SIFT):" << endl;

    Mat orbImg1 = imgCV1.clone();
    Mat orbImg2 = imgCV2.clone();

    Mat orbMatches;
    detectAndMatchORB(orbImg1, orbImg2, orbMatches);

    imwrite("output/orb_matches.jpg", orbMatches);
    cout << "Immagini con i match ORB salvate." << endl;

    Mat siftImg1 = imgCV1.clone();
    Mat siftImg2 = imgCV2.clone();

    Mat siftMatches;
    detectAndMatchSIFT(siftImg1, siftImg2, siftMatches);

    imwrite("output/sift_matches.jpg", siftMatches);
    cout << "Immagine con i match SIFT salvata." << endl;

    return 0;
}
