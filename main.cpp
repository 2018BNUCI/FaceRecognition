#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <direct.h>
#include "GaborFR.h"

using namespace std;
using namespace cv;


void OptRect(Rect& r) {
	int a = r.x, b = r.y, c = r.width, d = r.height, k = 7;
	r.x = (a - c / k) > 0 ? a - c / k : 0;
	r.y = (b - d / k) > 0 ? b - d / k : 0;
	r.width = c + c * 2 / k;
	r.height = d + d * 2 / k;
}

Mat imgRotate(Mat matSrc, float angle, bool direction)
{
	float theta = angle * CV_PI / 180.0;
	int nRowsSrc = matSrc.rows;
	int nColsSrc = matSrc.cols;
	// 如果是顺时针旋转
	if (!direction)
		theta = 2 * CV_PI - theta;
	// 全部以逆时针旋转来计算
	// 逆时针旋转矩阵
	float matRotate[3][3]{
		{ std::cos(theta), -std::sin(theta), 0 },
		{ std::sin(theta), std::cos(theta), 0 },
		{ 0, 0, 1 }
	};
	float pt[3][2]{
		{ 0, nRowsSrc },
		{ nColsSrc, nRowsSrc },
		{ nColsSrc, 0 }
	};
	for (int i = 0; i < 3; i++)
	{
		float x = pt[i][0] * matRotate[0][0] + pt[i][1] * matRotate[1][0];
		float y = pt[i][0] * matRotate[0][1] + pt[i][1] * matRotate[1][1];
		pt[i][0] = x;
		pt[i][1] = y;
	}
	// 计算出旋转后图像的极值点和尺寸
	float fMin_x = min(min(min(pt[0][0], pt[1][0]), pt[2][0]), (float)0.0);
	float fMin_y = min(min(min(pt[0][1], pt[1][1]), pt[2][1]), (float)0.0);
	float fMax_x = max(max(max(pt[0][0], pt[1][0]), pt[2][0]), (float)0.0);
	float fMax_y = max(max(max(pt[0][1], pt[1][1]), pt[2][1]), (float)0.0);
	int nRows = cvRound(fMax_y - fMin_y + 0.5) + 1;
	int nCols = cvRound(fMax_x - fMin_x + 0.5) + 1;
	int nMin_x = cvRound(fMin_x + 0.5);
	int nMin_y = cvRound(fMin_y + 0.5);
	// 拷贝输出图像
	Mat matRet(nRows, nCols, matSrc.type(), Scalar(0));
	for (int j = 0; j < nRows; j++)
	{
		for (int i = 0; i < nCols; i++)
		{
			// 计算出输出图像在原图像中的对应点的坐标，然后复制该坐标的灰度值
			// 因为是逆时针转换，所以这里映射到原图像的时候可以看成是，输出图像
			// 到顺时针旋转到原图像的，而顺时针旋转矩阵刚好是逆时针旋转矩阵的转置
			// 同时还要考虑到要把旋转后的图像的左上角移动到坐标原点。
			int x = (i + nMin_x) * matRotate[0][0] + (j + nMin_y) * matRotate[0][1];
			int y = (i + nMin_x) * matRotate[1][0] + (j + nMin_y) * matRotate[1][1];
			if (x >= 0 && x < nColsSrc && y >= 0 && y < nRowsSrc)
			{
				matRet.at<Vec3b>(j, i) = matSrc.at<Vec3b>(y, x);
			}
		}
	}
	return matRet;
}

int main() {
	// 获取图片
	Mat image = imread("C:/Users/User/Documents/test/1.jpg"), image_gray;
	if (image.empty()) {
		printf("文件不存在!");
		system("pause");
		return 0;
	}
	//imshow("原图", image);
	//waitKey(0);
	// 直方图均衡化
	
	Mat imageRGB[3];
	split(image, imageRGB);
	for (int i = 0; i < 3; i++) {
		equalizeHist(imageRGB[i], imageRGB[i]);
	}
	merge(imageRGB, 3, image);
	//imshow("直方图均衡化", image);
	//waitKey(0);
	
	// YCrCb
	/*
	int64 t1 = getTickCount();
	Mat image_YCrCb(image);
	Mat result1, tmp;
	Mat Y, Cr, Cb;
	vector<Mat> channels;
	image.copyTo(tmp);
	cvtColor(tmp, tmp, CV_BGR2YCrCb);
	split(tmp, channels);
	Y = channels.at(0);
	Cr = channels.at(1);
	Cb = channels.at(2);
	result1 = Mat::zeros(image.size(), CV_8UC1);
	int Max_row = -1, Max_col = -1, Min_row = -1, Min_col = -1;
	for (int i = 0; i < result1.rows; i++)
	{
		//各个图首行的指针
		uchar* currentCr = Cr.ptr< uchar>(i);
		uchar* currentCb = Cb.ptr< uchar>(i);
		uchar* current = result1.ptr< uchar>(i);
		for (int j = 0; j < result1.cols; j++)
		{
			if ((currentCr[j] >= 140) && (currentCr[j] <= 160) && (currentCb[j] >= 80) && (currentCb[j] <= 130)) {
				current[j] = 255;
				if (Max_row == -1)
					Max_row = i;
				else Max_row = max(i, Max_row);
				if (Max_col == -1)
					Max_col = j;
				else Max_col = max(j, Max_row);
				if (Min_row == -1)
					Min_row = i;
				else Min_row = min(i, Min_row);
				if (Min_col == -1)
					Min_col = j;
				else Min_col = min(j, Min_col);
			}
			else
				current[i] = 0;
		}
	}
	int64 t2 = getTickCount();
	cout << "YCrCb颜色空间：" << (t2 - t1) / getTickFrequency() << endl;
	rectangle(image_YCrCb, Rect(Min_row, Min_col, abs(Max_row - Min_row), abs(Max_col - Min_col)), Scalar(255, 0, 0), 1, LINE_8, 0);
	imshow("YCrCb", image_YCrCb);
	waitKey(0);
	*/
	// 将图像转成灰度图
	cvtColor(image, image_gray, CV_BGR2GRAY);
	equalizeHist(image_gray, image_gray);

	// 识别眼睛和面部
	CascadeClassifier eye_Classifier;
	CascadeClassifier face_cascade;

	if (!eye_Classifier.load("C:/Users/User/Documents/test/opencv/haarcascade_eye.xml")) {
		cout << "Load haarcascade_eye.xml failed" << endl;
		system("pause");
		return 0;
	}
	int64 kaishi = getTickCount();
	if (!face_cascade.load("C:/Users/User/Documents/test/opencv/haarcascade_frontalface_alt.xml")) {
		cout << "Load face_cascade failed!" << endl;
		system("pause");
		return 0;
	}

	// 面部矩形
	vector<Rect> faceRect;

	face_cascade.detectMultiScale(image_gray, faceRect, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	//cout << "faceRect.size " << (int)faceRect.size() << endl;
	Rect rr = faceRect[0];
	//cout << rr.height << " " << rr.width << " " << rr.x << " " << rr.y << endl;
	// 优化脸部截取矩形
	OptRect(faceRect[0]);
	rectangle(image, faceRect[0], Scalar(0, 0, 255));
	int64 jieshu = getTickCount();
	cout << "Haar特征: " << (jieshu - kaishi) / getTickFrequency() << endl;
	//imshow("Haar特征", image);
	//waitKey(0);
	
	
	// 将Mat的数据类型变为 CV_8UC3
	uchar *buf, *src;
	buf = new uchar[image.total()*image.elemSize()];
	src = image.data;
	for (int i = 0; i < image.cols; i++) {
		for (int j = 0; j < image.rows; j++) {
			buf[3 * (image.cols*j + i)] = src[3 * (image.cols*j + i)];
			buf[3 * (image.cols*j + i)+1] = src[3 * (image.cols*j + i)+1];
			buf[3 * (image.cols*j + i)+2] = src[3 * (image.cols*j + i)+2];
		}
	}
	// ans1: 原图+CV_8UC3
	Mat ans1(image.rows, image.cols, CV_8UC3, buf);
	// 截取人脸的大概位置
	Mat ans2(ans1, faceRect[0]);
	//imshow("ans1", ans1);
	//waitKey(0);
	Mat result, bgModel, fgModel, img2;
	
	grabCut(ans1, result, faceRect[0], bgModel, fgModel, 5, GC_INIT_WITH_RECT);
	//imshow("grab", result);
	//waitKey(0);
	compare(result, GC_PR_FGD, result, CMP_EQ);
	//imshow("result", result);
	//waitKey(0);
	Mat foreground(image.size(), CV_8UC3, Scalar(255, 255, 255));
	image.copyTo(foreground, result);//copyTo有两种形式，此形式表示result为mask
	//imshow("foreground", foreground);
	//waitKey(0);
	Mat fore = foreground(faceRect[0]);
	Mat fore_gray;
	cvtColor(fore, fore_gray, CV_BGR2GRAY);
	//imshow("fore_gray", fore_gray);
	//waitKey(0);
	cout << "here" << endl;
	vector<Rect> eyeRect;
	eye_Classifier.detectMultiScale(fore_gray, eyeRect, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));
	cout << "eyeRect.size " << (int)eyeRect.size() << endl;
	for (size_t i = 4; i < 6; i++) {
		rectangle(fore, eyeRect[i], Scalar(0, 0, 255));
	}
	cout << "hhh" << endl;
	imshow("fore",fore);
	waitKey(0);
	cout << "fuck" << endl;
	Mat foreRot;
	foreRot = imgRotate(fore, 10, true);
	imshow("foreRot", foreRot);
	waitKey(0);
	return 0;
}

