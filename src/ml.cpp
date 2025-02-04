#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Funzione per caricare i classificatori
bool loadClassifiers(const string &facePath, const string &eyesPath, const string &smilePath, 
                     CascadeClassifier &faceCascade, CascadeClassifier &eyesCascade, CascadeClassifier &smileCascade) {
    if (!faceCascade.load(facePath)) {
        cout << "Errore nel caricamento del classificatore facciale!" << endl;
        return false;
    }
    if (!eyesCascade.load(eyesPath)) {
        cout << "Errore nel caricamento del classificatore occhi!" << endl;
        return false;
    }
    if (!smileCascade.load(smilePath)) {
        cout << "Errore nel caricamento del classificatore sorriso!" << endl;
        return false;
    }
    return true;
}

// Funzione per rilevare le facce nell'immagine
void detectFaces(Mat &image, vector<Rect> &faces, CascadeClassifier &faceCascade) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));
}

// Funzione per rilevare gli occhi in una regione facciale
void detectEyes(Mat &faceROI, vector<Rect> &eyes, CascadeClassifier &eyesCascade) {
    Mat gray;
    cvtColor(faceROI, gray, COLOR_BGR2GRAY);
    eyesCascade.detectMultiScale(gray, eyes, 1.1, 4, 0, Size(30, 30));
}

// Funzione per rilevare il sorriso in una regione facciale
void detectSmile(Mat &faceROI, vector<Rect> &smiles, CascadeClassifier &smileCascade) {
    Mat gray;
    cvtColor(faceROI, gray, COLOR_BGR2GRAY);
    smileCascade.detectMultiScale(gray, smiles, 1.8, 4, 0, Size(30, 30));
}

// Funzione per disegnare i rettangoli per il viso, occhi e sorriso
void drawFaceFeatures(Mat &image, vector<Rect> &faces, CascadeClassifier &eyesCascade, CascadeClassifier &smileCascade) {
    for (size_t i = 0; i < faces.size(); i++) {
        // Disegna un rettangolo blu per il viso
        rectangle(image, faces[i], Scalar(255, 0, 0), 2); 

        // Estrai la regione del volto
        Mat faceROI = image(faces[i]);
        
        // Rileva gli occhi
        vector<Rect> eyes;
        detectEyes(faceROI, eyes, eyesCascade);

        // Disegna cerchi verdi per gli occhi
        for (size_t j = 0; j < eyes.size(); j++) {
            Point center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25); // Calcola il raggio del cerchio
            circle(image, center, radius, Scalar(0, 255, 0), 2); // Cerchio verde per gli occhi
        }

        // Rileva il sorriso
        vector<Rect> smiles;
        detectSmile(faceROI, smiles, smileCascade);

        // Disegna un rettangolo rosso per ogni sorriso rilevato
        for (size_t k = 0; k < smiles.size(); k++) {
            Rect smileRect(faces[i].x + smiles[k].x, faces[i].y + smiles[k].y, smiles[k].width, smiles[k].height);
            rectangle(image, smileRect, Scalar(0, 0, 255), 2); // Rettangolo rosso per il sorriso
        }
    }
}
