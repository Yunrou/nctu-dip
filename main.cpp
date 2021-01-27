#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <highgui.h>
#include "cv.h"
using namespace std;
using namespace cv;

Mat Method1(Mat);
Mat Method2(Mat);
Mat MyDFT(Mat, string);
Mat MyIDFT(Mat, string);
Mat MotionBlurFilterInSpatial(Mat);
Mat MotionBlurFilterInFrequency(Mat);

int main(){

	// Read input images
	// Fig3.tif is in openCV\bin\Release
	for (int i = 1; i < 5; i++)
	{
		string s = "Q" + to_string(i);
		Mat SrcImg = imread(s + ".tif", CV_LOAD_IMAGE_GRAYSCALE);
		imwrite(s + "_output_method1.tif", Method1(SrcImg));
		imwrite(s + "_output_method2.tif", Method2(SrcImg));
	}
	
	
	waitKey(0);
	return 0;
}
Mat Method1(Mat SrcImg)
{
	// h is filter in spatial domain
	Mat F = MyDFT(SrcImg, string("FTImg1"));
	Mat h = MotionBlurFilterInSpatial(SrcImg);
	Mat H = MyDFT(h, string("H for Method1"));

	Mat G;
	mulSpectrums(F, H, G, DFT_ROWS);

	Mat g = MyIDFT(G, string("g_method1"));
	g.convertTo(g, CV_8UC1, 255);
	float xoffset = (g.rows - SrcImg.rows) / 2;
	float yoffset = (g.cols - SrcImg.cols) / 2;
	g = g(Rect(yoffset, xoffset, SrcImg.cols, SrcImg.rows));
	imshow("Method1", g);

	return g;
}
Mat Method2(Mat SrcImg)
{
	Mat F = MyDFT(SrcImg, string("FTImg2"));
	Mat H = MotionBlurFilterInFrequency(SrcImg);
	// to see how different between method 1 and 2
	Mat h = MyIDFT(H, string("h"));
	h.convertTo(h, CV_8UC1, 255);
	imwrite("h in spatial for method2.tif", h);
	Mat G;
	
	mulSpectrums(F, H, G, DFT_ROWS);

	Mat g = MyIDFT(G, string("g_method2"));
	
	g.convertTo(g, CV_8UC1, 255);
	float xoffset = (g.rows - SrcImg.rows) / 2;
	float yoffset = (g.cols - SrcImg.cols) / 2;
	g = g(Rect(yoffset, xoffset, SrcImg.cols, SrcImg.rows));
	imshow("Method2", g);  

	return g;
}
Mat MyDFT(Mat SrcImg, string s)
{
	Mat paddedImg;
	int m = getOptimalDFTSize(SrcImg.rows);
	int n = getOptimalDFTSize(SrcImg.cols);
	n = max(m, n);
	// expand the border for efficiency
	copyMakeBorder(SrcImg, paddedImg, 0, n - SrcImg.rows, 0, n - SrcImg.cols, BORDER_REFLECT_101);

	Mat planes[] = { Mat_<float>(paddedImg), Mat::zeros(paddedImg.size(), CV_32F) };

	Mat complexImg;
	merge(planes, 2, complexImg);	// merge 2 channels
	dft(complexImg, complexImg, DFT_COMPLEX_OUTPUT);	// dft(input, output)
	
	// crop the spectrum, if it has an odd number of rows or columns
	complexImg = complexImg(Rect(0, 0, complexImg.cols & -2, complexImg.rows & -2));
	
	// let the origin in the center of img
	int cx = complexImg.cols / 2;
	int cy = complexImg.rows / 2;

	Mat q0(complexImg, Rect(0, 0, cx, cy));
	Mat q1(complexImg, Rect(cx, 0, cx, cy));
	Mat q2(complexImg, Rect(0, cy, cx, cy));
	Mat q3(complexImg, Rect(cx, cy, cx, cy));

	// exchange Top-Left with Bottom-Right & Top-Right with Bottom-Left
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	split(complexImg, planes);	// planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I)) 
	Mat magImg;
	magnitude(planes[0], planes[1], magImg);	// planes[0] = magnitude (input, input, output)

	magImg += Scalar::all(1);	// magImg = log(1+planes[0])
	log(magImg, magImg);

	normalize(magImg, magImg, 0, 1, CV_MINMAX);
	magImg.convertTo(magImg, CV_8UC1, 255);
	imwrite(s + ".tif", magImg);

	return complexImg;
}
Mat MyIDFT(Mat complexImg, string s)
{
	Mat idftImg;
	idft(complexImg, idftImg, DFT_REAL_OUTPUT);

	Mat planes[] = { Mat::zeros(complexImg.size(), CV_32F), Mat::zeros(complexImg.size(), CV_32F) };
	split(idftImg, planes); // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], idftImg); // sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
	normalize(idftImg, idftImg, 0, 1, NORM_MINMAX);

	return idftImg;
}
Mat MotionBlurFilterInSpatial(Mat SrcImg)
{
	int m = getOptimalDFTSize(SrcImg.rows);
	int n = getOptimalDFTSize(SrcImg.cols);
	n = max(m, n);
	Mat filter(n, n, CV_8UC1, Scalar::all(0));
	for (int i = 0; i < 0.1*n; i++)
		for (int j = 0; j < 0.1*n; j++)
			if (i == j)
				filter.at<uchar>(i, j) = 255;

	imwrite("h in spatial for method1.tif", filter);
	return filter;
}
Mat MotionBlurFilterInFrequency(Mat SrcImg)
{
	int m = getOptimalDFTSize(SrcImg.rows);
	int n =  getOptimalDFTSize(SrcImg.cols);
	n = max(m, n);
	Mat filter(n, n, CV_32F, Scalar::all(0));

	Mat planes[] = { Mat_<float>(filter), Mat::zeros(filter.size(), CV_32F) };
	Mat complexImg = Mat::zeros(filter.size(), CV_32F);

	int offset = (int)n / 2 + (int)n / 2;
	int pos;
	double theta, theta2, sinc;
	
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			pos = i + j - offset;
			theta = CV_PI*0.1*pos;
			// planes[0]: cos (Real), planes[1]: sin (Imaginary)
			if (pos == 0)
				sinc = 1;
			else
				sinc = sin(theta) / (theta);
				
			planes[0].at<float>(i, j) = cos(theta)*sinc;
			planes[1].at<float>(i, j) = -sin(theta)*sinc;
		}
	}

	merge(planes, 2, complexImg);

	Mat magImg;
	magnitude(planes[0], planes[1], magImg);	// planes[0] = magnitude (input, input, output)
	
	magImg += Scalar::all(1);	// magImg = log(1+planes[0])
	log(magImg, magImg);
	
	normalize(magImg, magImg, 0, 1, CV_MINMAX);
	magImg.convertTo(magImg, CV_8UC1, 255);
	imwrite("H for method2.tif", magImg);

	return complexImg;
}