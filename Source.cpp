#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

//vars of connected compo
Mat img;
int threshval = 100;

// image name from args
std::string image_file;


//-----------------------------------------------------
void show_histogram(std::string const& name, cv::Mat1b const& image)
{
    // Set histogram bins count
    int bins = 256;
    int histSize[] = { bins };
    // Set ranges for histogram bins
    float lranges[] = { 0, 256 };
    const float* ranges[] = { lranges };
    // create matrix for histogram
    Mat hist;
    int channels[] = { 0 };

    // create matrix for histogram visualization
    int const hist_height = 256;
    Mat3b hist_image = Mat3b::zeros(hist_height, bins);

    calcHist(&image, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);

    double max_val = 0;
    minMaxLoc(hist, 0, &max_val);

    // visualize each bin
    for (int b = 0; b < bins; b++) {
        float const binVal = hist.at<float>(b);
        int   const height = cvRound(binVal * hist_height / max_val);
        line
        (hist_image
            , Point(b, hist_height - height), Point(b, hist_height)
            , Scalar::all(255)
        );
    }
    imshow(name, hist_image);
}


//--------------------------------------------------
void get_args(int argc, const char* argv[]) {

    if (argc > 0) {

        image_file = argv[1];

        std::cout << "file : " << image_file << std::endl;

    }
    else {
        std::cout << "no args !!" << std::endl;
        exit(0);
    }

}

//-----------------------------------------------------
static void connectedCompo(int, void*, string s, int c)
{
    Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
    Mat labelImage(img.size(), CV_32S);
    int nLabels = connectedComponents(bw, labelImage, c);
    std::vector<Vec3b> colors(nLabels);
    colors[0] = Vec3b(0, 0, 0);//background
    for (int label = 1; label < nLabels; ++label) {
        colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
    }
    Mat dst(img.size(), CV_8UC3);
    for (int r = 0; r < dst.rows; ++r) {
        for (int c = 0; c < dst.cols; ++c) {
            int label = labelImage.at<int>(r, c);
            Vec3b& pixel = dst.at<Vec3b>(r, c);
            pixel = colors[label];
        }
    }

    Mat imagee;

    namedWindow("Connected Components (" + to_string(c) + " con) " + s, WINDOW_NORMAL);

    imshow("Connected Components ("+ to_string(c)+" con) "+s, dst);


    
}


//------------------------------
int main(int argc, const char* argv[])
{
    //----------get image file from args "107941.jpg" -----------------


    get_args(argc, argv);



    // here you can use cv::IMREAD_GRAYSCALE to load grayscale image, see image2
    Mat3b const image1 = cv::imread(image_file, cv::IMREAD_COLOR);
    Mat image1_gray;
    cvtColor(image1, image1_gray, cv::COLOR_BGR2GRAY);
    imshow("image1", image1_gray);
    show_histogram("image1 hist", image1_gray);


    int nbr_pixel = 0;
    vector<int> his(258);
    int a;

    for (int j = 0; j < image1_gray.rows; j++) { // on parcours notre image par rangs
        for (int i = 0; i < image1_gray.cols; i++) { // on parcout notre image par colone
            a = image1_gray.at<uchar>(j, i);
            his.at(a) = his.at(a) + 1; // on remplit l'histogramme de l'image image1_gray
        }
    }

    for (int i = 0; i < 256; i++) {
        nbr_pixel = nbr_pixel + his.at(i); //  le nombre de pixel
    }
    //
    vector<float> proba(256);
    for (int i = 0; i < 256; i++) {
        proba.at(i) = (float)his.at(i) / nbr_pixel; // step 3
        //cout<<proba.at(i)<<endl;
    }

    vector<float> Ubright(256);
    vector<float> Udark(256);
    int c;
    int best_c, best_a;
    float bright_proba = 0;
    float dark_proba = 0;
    float selectionght_proba = 0;
    float entropy = -40000;
    for (int a = 0; a < 255; a++) {
        for (int c = a + 1; c < 256; c++) {
            for (int x = 0; x < 256; x++) {
                if (x <= a) {
                    Ubright.at(x) = 0;
                    Udark.at(x) = 1;
                }
                else if (a < x && x < c) {
                    Ubright.at(x) = (float)(x - a) / (c - a);
                    Udark.at(x) = (float)(x - c) / (a - c);
                }
                else if (x >= c) {
                    Ubright.at(x) = 1;
                    Udark.at(x) = 0;
                }
            }
            for (int i = 0; i < 256; i++) {
                dark_proba = dark_proba + Udark.at(i) * proba.at(i);
                bright_proba = bright_proba + Ubright.at(i) * proba.at(i);
            }
            if (((0 - dark_proba) * log2(dark_proba) + (bright_proba * log2(bright_proba))) > entropy) {
                entropy = ((0 - dark_proba) * log2(dark_proba) + (bright_proba * log2(bright_proba)));
                best_c = c;
                best_a = a;
            }

        }
    }



    Mat3b const image11 = cv::imread(image_file, cv::IMREAD_COLOR);
    Mat image11_gray;
    cvtColor(image11, image11_gray, cv::COLOR_BGR2GRAY);
    for (int j = 0; j < image1_gray.rows; j++) {
        for (int i = 0; i < image1_gray.cols; i++) {
            image11_gray.at<uchar>(j, i) = image1_gray.at<uchar>(j, i);
        }
    }
    Mat3b const image12 = cv::imread(image_file, cv::IMREAD_COLOR);
    Mat image12_gray;
    cvtColor(image12, image12_gray, cv::COLOR_BGR2GRAY);
    for (int j = 0; j < image1_gray.rows; j++) {
        for (int i = 0; i < image1_gray.cols; i++) {
            image12_gray.at<uchar>(j, i) = image1_gray.at<uchar>(j, i);
        }
    }


    int nombre_pixel_zero_bestA = 0;
    for (int i = best_a; i < 256; i++) {
        nombre_pixel_zero_bestA = nombre_pixel_zero_bestA + his.at(i); //  le nombre de pixel
        //cout << nombre_pixel_zero_bestA<<endl;
    }


    vector<float> proba11(256);
    for (int i = best_a; i < 256; i++) {

        proba.at(i) = (float)his.at(i) / nombre_pixel_zero_bestA; // step 3

        cout << proba.at(i) << endl;
    }



    vector<float> Ubright11(256);
    vector<float> Udark11(256);
    int best_c11, best_a11;
    float bright_proba11 = 0;
    float dark_proba11 = 0;
    float entropy11 = -40000;
    for (int a = best_a; a < 255; a++) {
        for (int c = a + 1; c < 256; c++) {
            for (int x = best_a; x < 256; x++) {
                if (x <= a) {
                    Ubright11.at(x) = 0;
                    Udark11.at(x) = 1;
                }
                else if (a < x && x < c) {
                    Ubright11.at(x) = (float)(x - a) / (c - a);
                    Udark11.at(x) = (float)(x - c) / (a - c);
                }
                else if (x >= c) {
                    Ubright11.at(x) = 1;
                    Udark11.at(x) = 0;
                }
            }
            for (int i = best_a; i < 256; i++) {
                dark_proba11 = dark_proba11 + Udark11.at(i) * proba.at(i);
                bright_proba11 = bright_proba11 + Ubright11.at(i) * proba.at(i);
            }
            if (((0 - dark_proba11) * log2(dark_proba11) + (bright_proba11 * log2(bright_proba11))) > entropy11) {
                entropy11 = ((0 - dark_proba11) * log2(dark_proba11) + (bright_proba11 * log2(bright_proba11)));
                best_c11 = c;
                best_a11 = a;
                cout << "best c = " << best_c11 << endl;
                cout << "best a = " << best_a11 << endl;
            }

        }
    }

    for (int j = 0; j < image11_gray.rows; j++) {
        for (int i = 0; i < image11_gray.cols; i++) {
            if (image11_gray.at<uchar>(j, i) > best_c11)
                image11_gray.at<uchar>(j, i) = 255;
            else
                image11_gray.at<uchar>(j, i) = 0;

        }
    }
    for (int j = 0; j < image12_gray.rows; j++) {
        for (int i = 0; i < image12_gray.cols; i++) {
            if (image12_gray.at<uchar>(j, i) > best_a11)
                image12_gray.at<uchar>(j, i) = 255;
            else
                image12_gray.at<uchar>(j, i) = 0;

        }
    }



    Mat3b const image21 = cv::imread(image_file, cv::IMREAD_COLOR);
    Mat image21_gray;
    cvtColor(image21, image21_gray, cv::COLOR_BGR2GRAY);
    for (int j = 0; j < image1_gray.rows; j++) {
        for (int i = 0; i < image1_gray.cols; i++) {
            image21_gray.at<uchar>(j, i) = image1_gray.at<uchar>(j, i);
        }
    }
    Mat3b const image22 = cv::imread(image_file, cv::IMREAD_COLOR);
    Mat image22_gray;
    cvtColor(image22, image22_gray, cv::COLOR_BGR2GRAY);
    for (int j = 0; j < image1_gray.rows; j++) {
        for (int i = 0; i < image1_gray.cols; i++) {
            image22_gray.at<uchar>(j, i) = image1_gray.at<uchar>(j, i);
        }
    }



    int nombre_pixel_zero_bestC = 0;
    for (int i = 0; i < best_c; i++) {
        nombre_pixel_zero_bestC = nombre_pixel_zero_bestC + his.at(i); //  le nombre de pixel
        //cout << nombre_pixel_zero_bestA<<endl;
    }


    vector<float> proba21(256);
    for (int i = best_a; i < 256; i++) {

        proba.at(i) = (float)his.at(i) / nombre_pixel_zero_bestC; // step 3

        cout << proba.at(i) << endl;
    }




    vector<float> Ubright21(256);
    vector<float> Udark21(256);
    int best_c21, best_a21;
    float bright_proba21 = 0;
    float dark_proba21 = 0;
    float entropy21 = -40000;
    for (int a = 0; a < best_c - 1; a++) {
        for (int c = a + 1; c < best_c; c++) {
            for (int x = 0; x < best_c; x++) {
                if (x <= a) {
                    Ubright21.at(x) = 0;
                    Udark21.at(x) = 1;
                }
                else if (a < x && x < c) {
                    Ubright21.at(x) = (float)(x - a) / (c - a);
                    Udark21.at(x) = (float)(x - c) / (a - c);
                }
                else if (x >= c) {
                    Ubright21.at(x) = 1;
                    Udark21.at(x) = 0;
                }
            }
            for (int i = 0; i < best_c; i++) {
                dark_proba21 = dark_proba21 + Udark21.at(i) * proba.at(i);
                bright_proba21 = bright_proba21 + Ubright21.at(i) * proba.at(i);
            }
            if (((0 - dark_proba21) * log2(dark_proba21) + (bright_proba21 * log2(bright_proba21))) > entropy21) {
                entropy21 = ((0 - dark_proba21) * log2(dark_proba21) + (bright_proba21 * log2(bright_proba21)));
                best_c21 = c;
                best_a21 = a;
                cout << "best c = " << best_c21 << endl;
                cout << "best a = " << best_a21 << endl;
            }

        }
    }



    for (int j = 0; j < image21_gray.rows; j++) {
        for (int i = 0; i < image21_gray.cols; i++) {
            if (image21_gray.at<uchar>(j, i) > best_c21)
                image21_gray.at<uchar>(j, i) = 255;
            else
                image21_gray.at<uchar>(j, i) = 0;

        }
    }
    for (int j = 0; j < image22_gray.rows; j++) {
        for (int i = 0; i < image22_gray.cols; i++) {
            if (image22_gray.at<uchar>(j, i) > best_a21)
                image22_gray.at<uchar>(j, i) = 255;
            else
                image22_gray.at<uchar>(j, i) = 0;

        }
    }

    //------------------------------
    img = image11_gray;
    namedWindow("Connected Components (4 con) 11", WINDOW_AUTOSIZE);
    connectedCompo(threshval, 0, "11", 4);

    
    //------------------------------

     //------------------------------
    img = image12_gray;
    namedWindow("Connected Components (4 con) 12", WINDOW_AUTOSIZE);
    connectedCompo(threshval, 0, "12", 4);

    
    //------------------------------

     //------------------------------
    img = image21_gray;
    namedWindow("Connected Components (4 con) 21", WINDOW_AUTOSIZE);
    connectedCompo(threshval, 0, "21", 4);

    
    //------------------------------

     //------------------------------
    img = image22_gray;
    namedWindow("Connected Components (4 con) 22", WINDOW_AUTOSIZE);
    connectedCompo(threshval, 0, "22", 4);

    
    //------------------------------



    imwrite("Gray_Image11_"+ image_file, image11_gray);
    imwrite("Gray_Image12_" + image_file, image12_gray);
    imwrite("Gray_Image21_" + image_file, image21_gray);
    imwrite("Gray_Image22_" + image_file, image22_gray);
    imshow("image11", image11_gray);
    imshow("image12", image12_gray);
    imshow("image21", image21_gray);
    imshow("image22", image22_gray);

    waitKey();
    return 0;
}