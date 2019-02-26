#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void OptRect(Rect& r) {
	int a = r.x, b = r.y, c = r.width, d = r.height, k = 7;
	r.x = (a - c / k) > 0 ? a - c / k : 0;
	r.y = (b - d / k) > 0 ? b - d / k : 0;
	r.width = c + c * 2 / k;
	r.height = d + d * 2 / k;
}

int main() {
	Mat image = imread("C:/Users/User/Documents/test/1.jpg"), image_gray;
	if (image.empty()) {
		printf("文件不存在!");
		system("pause");
		return 0;
	}
	/*namedWindow("p1", CV_WINDOW_NORMAL);
	imshow("p1", image);
	waitKey(0);
	*/
	cvtColor(image, image_gray, CV_BGR2GRAY);
	equalizeHist(image_gray, image_gray);

	/*
	namedWindow("p1_gray", CV_WINDOW_NORMAL);
	imshow("p1_gray", image_gray);
	waitKey(0);
	*/
	CascadeClassifier eye_Classifier;
	CascadeClassifier face_cascade;

	if (!eye_Classifier.load("C:/Users/User/Documents/test/opencv/haarcascade_eye.xml")) {
		cout << "Load haarcascade_eye.xml failed" << endl;
		system("pause");
		return 0;
	}
	if (!face_cascade.load("C:/Users/User/Documents/test/opencv/haarcascade_frontalface_alt.xml")) {
		cout << "Load face_cascade failed!" << endl;
		system("pause");
		return 0;
	}
	vector<Rect> faceRect;

	face_cascade.detectMultiScale(image_gray, faceRect, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	cout << "faceRect.size " << (int)faceRect.size() << endl;
	Rect rr = faceRect[0];
	cout << rr.height << " " << rr.width << " " << rr.x << " " << rr.y << endl;
	OptRect(faceRect[0]);
	rectangle(image, faceRect[0], Scalar(0, 0, 255));
	imshow("resultWithRectangle", image);
	waitKey(0);
	

	/*rectangle(image, Rect(569, 823 , 100, 100), Scalar(0,0,255));*/

	uchar *buf, *src;
	buf = new uchar[image.total()*image.elemSize()];
	src = image.data;

	// 产生一个Buffer
	for (int i = 0; i < image.cols; i++) {
		for (int j = 0; j < image.rows; j++) {
			buf[3 * (image.cols*j + i)] = src[3 * (image.cols*j + i)];
			buf[3 * (image.cols*j + i)+1] = src[3 * (image.cols*j + i)+1];
			buf[3 * (image.cols*j + i)+2] = src[3 * (image.cols*j + i)+2];
		}
	}

	Mat ans1(image.rows, image.cols, CV_8UC3, buf);

	Mat ans2(ans1, faceRect[0]);
	imshow("ans1", ans1);
	waitKey(0);
	Mat result, bgModel, fgModel, img2;
	
	grabCut(ans1, result, faceRect[0], bgModel, fgModel, 5, GC_INIT_WITH_RECT);
	imshow("grab", result);
	waitKey(0);
	compare(result, GC_PR_FGD, result, CMP_EQ);
	imshow("result", result);
	waitKey(0);
	Mat foreground(image.size(), CV_8UC3, Scalar(255, 255, 255));
	image.copyTo(foreground, result);//copyTo有两种形式，此形式表示result为mask
	imshow("foreground", foreground);
	waitKey(0);
	Mat fore = foreground(faceRect[0]);
	Mat fore_gray;
	cvtColor(fore, fore_gray, CV_BGR2GRAY);
	imshow("fore_gray", fore_gray);
	waitKey(0);
	vector<Rect> eyeRect;
	eye_Classifier.detectMultiScale(fore_gray, eyeRect, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));
	cout << "eyeRect.size " << (int)eyeRect.size() << endl;
	for (size_t i = 0; i < eyeRect.size(); i++) {
		rectangle(fore, eyeRect[i], Scalar(0, 0, 255));
	}
	imshow("fore",fore);
	waitKey(0);
	return 0;
}