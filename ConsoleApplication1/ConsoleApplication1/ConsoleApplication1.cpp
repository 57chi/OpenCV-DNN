// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace dnn;
using namespace std;

int main()
{
	Mat img = imread("img.jpg");
	const String strModel = "frozen_inference_graph.pb";
	const String strConfig = "graph.pbtxt";
	Net net = readNetFromTensorflow(strModel, strConfig);

	resize(img, img, cv::Size(300, 300));

	int rows = img.rows;
	int cols = img.cols;

	Mat blob;
	blob = blobFromImage(img, (1.0), Size(rows, cols), Scalar(0, 0, 0), true, false);

	net.setInput(blob);

	Mat mat;

	mat = net.forward();

	Mat newMat;
	//newMat = mat.reshape(1, (int)mat.total() / mat.size[3]);
	newMat = mat.reshape(1, mat.size[2]);

	for (int i = 0; i < newMat.rows; i++)
	{
		float* data = newMat.ptr<float>(i, 0);
		float score = data[2];
		if (score > 0.2)
		{
			int left = data[3] * cols;
			int top = data[4] * rows;
			int right = data[5] * cols;
			int bottom = data[6] * rows;
			rectangle(img, Point(int(left), int(top)), Point(int(right), int(bottom)), (0, 0, 255), 2);
			cout << data[1] << endl;;
		}
	}

	imshow("img", img);
	waitKey();

	return 0;


}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
