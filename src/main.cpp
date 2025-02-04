#include "shi-tomasi.h"
#include "ml.h"
#include "sift.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Path per le immagini da leggere 
    string imagePath1 = "pano/selfie/s1.jpg";
    string imagePath2 = "pano/selfie/s2.jpg";

    Mat img1 = imread(imagePath1, IMREAD_COLOR);
    Mat img2 = imread(imagePath2, IMREAD_COLOR);

    // Check sulle immagini non vuote
    if (img1.empty() || img2.empty()) {
        cout << "Errore nel caricamento delle immagini!" << endl;
        return -1;
    }

    // Vettori per i corner rilevati nelle due immagini
    vector<Point2f> corners1, corners2;

    // Ridimensiona le immagini per garantire un confronto preciso
    Mat resizedImg1, resizedImg2;
    resizeImage(img1, resizedImg1, 510, 680);
    resizeImage(img2, resizedImg2, 510, 680);

    // Applico Shi-Tomasi Corner Detection su entrambe le immagini per il rilevamento dei corner
    Mat img_corners1, img_corners2;
    detectShiTomasiCorners(resizedImg1, img_corners1, corners1);
    detectShiTomasiCorners(resizedImg2, img_corners2, corners2);

    // Evidenzio quanti corner sono stati rilevati nelle immagini
    cout << "Numero di corner rilevati nella prima immagine: " << corners1.size() << endl;
    cout << "Numero di corner rilevati nella seconda immagine: " << corners2.size() << endl;

    // Salva le immagini con i corner rilevati
    string outputCorners1 = "output/shi_tomasi_corners1.jpg";
    string outputCorners2 = "output/shi_tomasi_corners2.jpg";
    imwrite(outputCorners1, img_corners1);
    imwrite(outputCorners2, img_corners2);
    cout << "Immagini con i corner salvate in: " << outputCorners1 << " e " << outputCorners2 << endl;

    // Variabili per la gestione dei match e la percentuale di matching 
    Mat result;
    int matchCount = 0, unmatchedCount = 0;

    // Confronta i corner tra le due immagini
    matchCorners(corners1, corners2, resizedImg1, resizedImg2, result, matchCount, unmatchedCount);

    // Salva l'immagine con i match
    string outputMatch = "output/shi_tomasi_matches.jpg";
    imwrite(outputMatch, result);

    cout << "Immagine con i match salvata in: " << outputMatch << endl;

    // **Chiamata alla funzione detectAndMatchSIFT**
    Mat imgMatches;  // Matrice per salvare l'immagine con i match
    detectAndMatchSIFT(resizedImg1, resizedImg2, imgMatches);

    // Salva l'immagine con i keypoints matching
    string outputMatches = "output/sift_matches.jpg";
    imwrite(outputMatches, imgMatches);
    cout << "Immagine con i match SIFT salvata in: " << outputMatches << endl;

    // Path per i file Haar
    string face_classifier_path = "opencv/data/haarcascades/haarcascade_frontalface_alt2.xml";
    string eyes_classifier_path = "opencv/data/haarcascades/haarcascade_eye.xml";
    string smile_classifier_path = "opencv/data/haarcascades/haarcascade_smile.xml";

    // Carica i classificatori
    CascadeClassifier faceCascade, eyesCascade, smileCascade;
    if (!loadClassifiers(face_classifier_path, eyes_classifier_path, smile_classifier_path, faceCascade, eyesCascade, smileCascade)) {
        return -1; // Se il caricamento del classificatore fallisce, esci
    }

    // Rileva faccia e caratteristiche facciali per la prima immagine
    vector<Rect> faces1;
    detectFaces(resizedImg1, faces1, faceCascade);

    // Disegna tratti facciali (viso, occhi e sorriso) nella prima immagine
    drawFaceFeatures(resizedImg1, faces1, eyesCascade, smileCascade);

    // Salva l'immagine con i tratti facciali rilevati
    string outputFace1 = "output/face_features1.jpg";
    imwrite(outputFace1, resizedImg1);
    cout << "Immagine con tratti facciali salvata in: " << outputFace1 << endl;

    // Rileva faccia e caratteristiche facciali per la seconda immagine
    vector<Rect> faces2;
    detectFaces(resizedImg2, faces2, faceCascade);

    // Disegna tratti facciali (viso, occhi e sorriso) nella seconda immagine
    drawFaceFeatures(resizedImg2, faces2, eyesCascade, smileCascade);

    // Salva l'immagine con i tratti facciali rilevati
    string outputFace2 = "output/face_features2.jpg";
    imwrite(outputFace2, resizedImg2);
    cout << "Immagine con tratti facciali salvata in: " << outputFace2 << endl;

    return 0;

}
