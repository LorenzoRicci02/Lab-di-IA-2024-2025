#ifndef ML_H
#define ML_H

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;

bool loadClassifiers(const string &facePath, const string &eyesPath, const string &smilePath, CascadeClassifier &faceCascade, CascadeClassifier &eyesCascade, CascadeClassifier &smileCascade);

void detectFaces(Mat &image, vector<Rect> &faces, CascadeClassifier &faceCascade);

void detectEyes(Mat &faceROI, vector<Rect> &eyes, CascadeClassifier &eyesCascade);

void detectSmile(Mat &faceROI, vector<Rect> &smiles, CascadeClassifier &smileCascade);

void drawFaceFeatures(Mat &image, vector<Rect> &faces, CascadeClassifier &eyesCascade, CascadeClassifier &smileCascade);

#endif 
