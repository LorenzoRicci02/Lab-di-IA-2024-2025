#include "ocv_corners.h"
#include "my_corners.h"
#include "ocv_orb.h"
#include "ocv_sift.h"
#include "my_brief.h"
#include "my_sift.h"
#include "my_orb.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono> 

using namespace cv;
using namespace std;
using namespace chrono;

int main() {
    // Percorsi per le immagini da utilizzare nella parte di Computer Vision
    string imagePathCV1 = "pano/celeb/lm.jpg";
    string imagePathCV2 = "pano/flip/lm_totalflip.jpg";

    // Carica le immagini per la parte di Computer Vision
    Mat imgCV1 = imread(imagePathCV1, IMREAD_COLOR);
    Mat imgCV2 = imread(imagePathCV2, IMREAD_COLOR);

    // Controllo se le immagini non sono vuote
    if (imgCV1.empty() || imgCV2.empty()) {
        cout << "Errore nel caricamento delle immagini per Computer Vision!" << endl;
        return -1;
    }

    // Parte Corner Detection
    cout << "- Parte Corner Detection (Shi-Tomasi, FAST, Harris):" << endl;


    /*/ PARTE SHI-TOMASI OPENCV /*/

    // Misura il tempo per la funzione detectShiTomasiCorners (OCV)
    Mat imgCorners1_1 = imgCV1.clone();
    vector<Point2f> corners1_1;
    auto start = high_resolution_clock::now();
    detectShiTomasiCorners(imgCorners1_1, imgCorners1_1, corners1_1);
    auto stop = high_resolution_clock::now();

    // Salvo l'immagine
    imwrite("output/ocv_shi1.jpg", imgCorners1_1);
    auto duration1 = duration_cast<milliseconds>(stop - start);
    cout << "\nTempo di esecuzione per Shi-Tomasi con OpenCV: " << duration1.count() << " ms" << endl;
    cout << "Corner rilevati con Shi-Tomasi (OCV): " << corners1_1.size() << endl;
    cout << "Immagine con i corner Shi-Tomasi (OCV) salvata." << endl;    


    /*/ PARTE SHI-TOMASI REIMPLEMENTATO /*/

    // Misura il tempo per la funzione ShiTomasiCorners (Mio)
    Mat imgCorners1_2 = imgCV1.clone();
    start = high_resolution_clock::now();
    vector<KeyPoint> corners1_2 = ShiTomasiCorners(imgCorners1_2, 0.08, 1000, 7);
    stop = high_resolution_clock::now();

    // Disegna i corner rilevati
    Mat dst1 = imgCorners1_2.clone();
    for (const auto& pt : corners1_2) {
        circle(dst1, pt.pt, 3, Scalar(0, 0, 255), FILLED);
    }
    imwrite("output/my_shi1.jpg", dst1);
    auto duration2 = duration_cast<milliseconds>(stop - start);
    cout << "Tempo di esecuzione per Shi-Tomasi reimplementato da me: " << duration2.count() << " ms" << endl;
    cout << "Corner rilevati Shi-Tomasi reimplementato: " << corners1_2.size() << endl;
    cout << "Immagine con i corner Shi-Tomasi salvata." << endl;

    // Sovrapposizione delle immagini (Shi-Tomasi OCV + Mio)
    Mat img1 = imread("output/ocv_shi1.jpg");
    Mat img2 = imread("output/my_shi1.jpg");

    Mat result1;
    addWeighted(img1, 0.5, img2, 0.5, 0, result1);

    // Salva il risultato della sovrapposizione
    imwrite("output/sovrapposizione_shi.jpg", result1);

    cout << "Immagini sovrapposte salvate.\n" << endl;


    /*/ PARTE FAST OPENCV /*/

    Mat imgFAST1 = imgCV1.clone();
    vector<KeyPoint> fastCorners1;
    start = high_resolution_clock::now();
    detectFASTCorners(imgFAST1, imgFAST1, fastCorners1);
    stop = high_resolution_clock::now();
    imwrite("output/ocv_fast1.jpg", imgFAST1);
    auto duration3 = duration_cast<milliseconds>(stop - start);
    cout << "Tempo di esecuzione per FAST con OpenCV: " << duration3.count() << " ms" << endl;
    cout << "Corner rilevati FAST (OCV): " << fastCorners1.size() << endl;
    cout << "Immagine con i corner FAST (OCV) salvata." << endl;


    /*/ PARTE FAST REIMPLEMENTATO /*/

    Mat imgCorners1_3 = imgCV1.clone();
    start = high_resolution_clock::now();
    vector<KeyPoint> corners1_3 = FASTCorners(imgCorners1_3, 50, true);
    stop = high_resolution_clock::now();

    // Disegna i corner sulla prima immagine
    Mat dst3 = imgCorners1_3.clone();
    for (const auto& pt : corners1_3) {
        circle(dst3, pt.pt, 3, Scalar(0, 0, 255), FILLED);
    }
    imwrite("output/my_fast1.jpg", dst3);
    auto duration4 = duration_cast<milliseconds>(stop - start);
    cout << "Tempo di esecuzione per FAST reimplementato da me: " << duration4.count() << " ms" << endl;
    cout << "Corner rilevati Fast reimplementato: " << corners1_3.size() << endl;
    cout << "Immagine con i corner FAST salvata." << endl;

    // Sovrapposizione delle immagini
    Mat img3 = imread("output/ocv_fast1.jpg");
    Mat img4 = imread("output/my_fast1.jpg");

    // Sovrapponi le immagini
    Mat result2;
    addWeighted(img3, 0.5, img4, 0.5, 0, result2);

    // Salva il risultato della sovrapposizione
    imwrite("output/sovrapposizione_fast.jpg", result2);

    cout << "Immagini sovrapposte salvate.\n" << endl;

    Mat imgHarris1 = imgCV1.clone();
    Mat result3;


    /*/ PARTE HARRIS OPENCV /*/

    vector<KeyPoint> harrisOCVKeypoints;
    start = high_resolution_clock::now();
    detectHarrisCorners(imgHarris1, result3, 3, 3, 0.04, 80, 8, harrisOCVKeypoints);
    stop = high_resolution_clock::now();
    auto durationHarris = duration_cast<milliseconds>(stop - start);

    imwrite("output/ocv_harris.jpg", result3);
    cout << "Tempo di esecuzione per Harris con OpenCV: " << durationHarris.count() << " ms" << endl;
    cout << "Corner rilevati Harris (OCV): " << harrisOCVKeypoints.size() << endl;
    cout << "Immagini con i corner Harris (OCV) salvata." << endl;


    /*/ PARTE HARRIS REIMPLEMENTATO /*/

    Mat imgHarris2 = imgCV1.clone();

    start = high_resolution_clock::now();
    vector<KeyPoint> keypoints = HarrisCorners(imgHarris2, 5, 0.2);
    stop = high_resolution_clock::now();

    Mat result4 = imgHarris2.clone();
    for (const auto& kp : keypoints) {
        circle(result4, kp.pt, 3, Scalar(0, 0, 255), FILLED);
    }

    auto durationHarris2 = duration_cast<milliseconds>(stop - start);

    imwrite("output/my_harris.jpg", result4);
    cout << "Tempo di esecuzione per Harris reimplementato da me: " << durationHarris2.count() << " ms" << endl;
    cout << "Corner rilevati Harris reimplementato: " << keypoints.size() << endl;
    cout << "Immagine con i corner Harris salvata." << endl;

    // Sovrapposizione delle immagini
    Mat img5 = imread("output/my_harris.jpg");
    Mat img6 = imread("output/ocv_harris.jpg");

    // Sovrapponi le immagini
    Mat result5;
    addWeighted(img5, 0.5, img6, 0.5, 0, result5);

    // Salva il risultato della sovrapposizione
    imwrite("output/sovrapposizione_harris.jpg", result5);

    cout << "Immagini sovrapposte salvate.\n" << endl;

    // Parte Descriptor Matching
    cout << "- Parte Descriptor Matching (ORB, SIFT):" << endl;


    // PARTE ORB OCV //

    Mat orbImg1 = imgCV1.clone();
    Mat orbImg2 = imgCV2.clone();

    Mat orbMatches;
    start = high_resolution_clock::now();
    detectAndMatchORB(orbImg1, orbImg2, orbMatches);
    stop = high_resolution_clock::now();
    auto durationOCVORB = duration_cast<milliseconds>(stop - start);

    imwrite("output/ocv_orb.jpg", orbMatches);
    cout << "Tempo di esecuzione ORB con OpenCV: " << durationOCVORB.count() << " ms" << endl;
    cout << "Immagini con i match ORB (OCV) salvate.\n" << endl;


    // PARTE ORB REIMPLEMENTATO //
    
    Mat myORB1 = imgCV1.clone();
    Mat myORB2 = imgCV2.clone();
    
    vector<KeyPoint> myKeypoints1, myKeypoints2;
    Mat myDescriptors1, myDescriptors2;
    vector<DMatch> myMatches;
    
    start = high_resolution_clock::now();
    computeORB(myORB1, myORB2, myKeypoints1, myKeypoints2, myDescriptors1, myDescriptors2, myMatches);
    stop = high_resolution_clock::now();
    
    Mat myOrbOutput;
    drawMatches(myORB1, myKeypoints1, myORB2, myKeypoints2, myMatches, myOrbOutput, Scalar(0, 0, 255), Scalar(0, 0, 255));
    
    imwrite("output/my_orb.jpg", myOrbOutput);
    
    auto duration_myorb = duration_cast<milliseconds>(stop - start);
    
    cout << "Tempo di esecuzione ORB reimplementato da me: " << duration_myorb.count() << " ms" << endl;
    cout << "Immagini con i match ORB reimplementato da me salvate." << endl;

    // Sovrapposizione tra ORB (OCV) e ORB (mio)
    Mat orbMatchesImg = imread("output/ocv_orb.jpg");
    Mat myOrbMatchesImg = imread("output/my_orb.jpg");

    Mat orbOverlay;
    addWeighted(orbMatchesImg, 0.5, myOrbMatchesImg, 0.5, 0, orbOverlay);
    imwrite("output/sovrapposizione_orb.jpg", orbOverlay);
    cout << "Immagini sovrapposte salvate." << endl;
    
    /*
    Mat siftImg1 = imgCV1.clone();
    Mat siftImg2 = imgCV2.clone();

    Mat siftMatches;
    detectAndMatchSIFT(siftImg1, siftImg2, siftMatches);

    imwrite("output/ocv_sift.jpg", siftMatches);
    cout << "Immagine con i match SIFT salvata." << endl;
    */
    return 0;
}
