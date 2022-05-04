// OpenCVApplication.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include "common.h"
using namespace cv;
using namespace std;
RNG rng(12345);

struct images {
	Mat_<uchar> img_bw;
	Mat_<uchar> negative_image;
	Mat_<uchar> contour_image;
};

const float MAXTHINNESS = 0.7;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void detectWhitePointsUsingHSV() {
	// HSV range to detect white/black color
	int minHue = 0, maxHue = 0;
	int minSat = 0, maxSat = 0;
	int minVal = 0, maxVal = 255;

	//// 1. Create mask settings UI with default HSV range to detect white color
	auto const MASK_WINDOW = "Mask Settings";
	namedWindow(MASK_WINDOW, CV_WINDOW_AUTOSIZE);

	// Create trackbars of mask settings window
	cvCreateTrackbar("Min Hue", MASK_WINDOW, &minHue, 0);
	cvCreateTrackbar("Max Hue", MASK_WINDOW, &maxHue, 179);
	cvCreateTrackbar("Min Sat", MASK_WINDOW, &minSat, 0);
	cvCreateTrackbar("Max Sat", MASK_WINDOW, &maxSat, 255);
	cvCreateTrackbar("Min Val", MASK_WINDOW, &minVal, 0);
	cvCreateTrackbar("Max Val", MASK_WINDOW, &maxVal, 255);

	while (true) {
		//// 2. Read and convert image to HSV color space
		Mat inputImage{ imread("Images/image010-40x (5).tif", IMREAD_COLOR) };
		Mat inputImageHSV;
		cvtColor(inputImage, inputImageHSV, COLOR_BGR2HSV);

		//// 3. Create mask and result (masked) image
		Mat mask;
		// params: input array, lower boundary array, upper boundary array, output array
		inRange(
			inputImageHSV,
			cv::Scalar(minHue, minSat, minVal),
			cv::Scalar(maxHue, maxSat, maxVal),
			mask
		);
		Mat resultImage;
		// params: src1    array, src2 array, output array, mask
		bitwise_and(inputImage, inputImage, resultImage, mask);

		//// 4. Show images
		imshow("Input Image", inputImage);
		imshow("Result (Masked) Image", resultImage);
		// imshow("Mask", mask);

		waitKey(0);
	}
}

void detectBlackPointsUsingHSV() {
	// HSV range to detect white/black color
	int minHue = 0, maxHue = 50;
	int minSat = 0, maxSat = 50;
	int minVal = 0, maxVal = 100;

	//// 1. Create mask settings UI with default HSV range to detect white color
	auto const MASK_WINDOW = "Mask Settings";
	namedWindow(MASK_WINDOW, CV_WINDOW_AUTOSIZE);

	// Create trackbars of mask settings window
	cvCreateTrackbar("Min Hue", MASK_WINDOW, &minHue, 0);
	cvCreateTrackbar("Max Hue", MASK_WINDOW, &maxHue, 360);
	cvCreateTrackbar("Min Sat", MASK_WINDOW, &minSat, 0);
	cvCreateTrackbar("Max Sat", MASK_WINDOW, &maxSat, 255);
	cvCreateTrackbar("Min Val", MASK_WINDOW, &minVal, 0);
	cvCreateTrackbar("Max Val", MASK_WINDOW, &maxVal, 50);

	while (true) {
		//// 2. Read and convert image to HSV color space
		Mat inputImage{ cv::imread("Images/40X (3).tif", IMREAD_COLOR) };
		Mat inputImageHSV;
		cvtColor(inputImage, inputImageHSV, cv::COLOR_BGR2HSV);//trasforma din RGB in HSV

		//// 3. Create mask and result (masked) image
		Mat mask;
		// params: input array, lower boundary array, upper boundary array, output array

		inRange(
			inputImageHSV,
			cv::Scalar(minHue, minSat, minVal),
			cv::Scalar(maxHue, maxSat, maxVal),
			mask
		);
		Mat resultImage;
		// params: src1    array, src2 array, output array, mask
		bitwise_and(inputImage, inputImage, resultImage, mask);

		//// 4. Show images
		imshow("Input Image", inputImage);
		imshow("Result (Masked) Image", resultImage);
		// imshow("Mask", mask);

		waitKey(0);
	}
}

int* computeHistogram(Mat_<uchar> img, int nr_bins, int* hist, float* pdf) {
	for (int i = 0; i <= nr_bins; i++) {
		hist[i] = 0;
	}

	int nr = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			nr = img(i, j);
			hist[nr]++;
		}
	}
	for (int i = 0; i < 256; i++) {
		pdf[i] = (float)hist[i] / (img.rows * img.cols);
	}

	return hist;
}

Mat_<uchar> binarization(Mat_<uchar> img) {
	float* pdf = (float*)malloc(256 * sizeof(float));
	int* hist = (int*)malloc(256 * sizeof(int));
	hist = computeHistogram(img, 256, hist, pdf);
	float Imax = img(0, 0);
	float Imin = img(0, 0);
	//intensitatea maxima
	for (int i = 255; i >= 0; i--) {
		if (hist[i] != 0) {
			Imax = i;
			break;
		}
	}
	//intensitatea minima
	for (int i = 0; i < 256; i++) {
		if (hist[i] != 0) {
			Imin = i;
			break;
		}
	}
	//pragul T:
	int T = (Imin + Imax) / 2;
	int k = 0;
	while (k = 0) {
		float N1 = 0;
		float N2 = 0;
		float termen1 = 0;
		float termen2 = 0;
		for (int g = Imin; g <= T; g++) {
			termen1 += g * hist[g];
		}
		for (int g = T + 1; g <= Imax; g++) {
			termen2 += g * hist[g];
		}
		for (int g = Imin; g <= T; g++) {
			N1 += hist[g];
		}
		for (int g = T + 1; g <= Imax; g++) {
			N2 += hist[g];
		}
		float G1 = termen1 / N1;
		float G2 = termen2 / N2;

		int Tk = (G1 + G2) / 2;

		if (abs(Tk - T) < 0.1) {
			k = 1;
		}
		T = Tk;
	}
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) < T)
				img(i, j) = 0;
			else
				img(i, j) = 255;
		}
	}
	//imshow("Imaginea binarizata", img);
	waitKey(0);
	return(img);
}


Mat exploreColorSpaces(Mat initialImage)
{
	//deschidem imaginea color
	char fname[MAX_PATH];
	//Mat initialImage;
	Mat imgHSV;
	Mat imgRGBA;
	Mat imgGRAY;
	Mat imgLAB;
	Mat imgXYZ;
	Mat imgYUV;
	Mat  imgHLS;
	Mat  imgLUV;
	Mat  imgYCrCb;
	Mat_<uchar> imgBin;
	Mat rgbchannel9[3];
	//while (openFileDlg(fname)) {
		//initialImage = imread(fname, IMREAD_COLOR);
		//RGB-->GRAYSCALE
	cvtColor(initialImage, imgGRAY, cv::COLOR_RGB2BGRA);
	//RGB-->RGBA
	cvtColor(initialImage, imgRGBA, cv::COLOR_BGR2RGBA);
	//RGB-->HSV
	cvtColor(initialImage, imgHSV, cv::COLOR_RGB2HSV);
	//RGB-->Lab
	cvtColor(initialImage, imgLAB, cv::COLOR_RGB2Lab);
	//RGB-->XYZ
	cvtColor(initialImage, imgXYZ, cv::COLOR_RGB2XYZ);
	//RGB-->YUV
	cvtColor(initialImage, imgYUV, cv::COLOR_RGB2YUV);
	//RGB-->HL
	cvtColor(initialImage, imgHLS, cv::COLOR_RGB2HLS);
	//RGB-->Luv
	cvtColor(initialImage, imgLUV, cv::COLOR_RGB2Luv);
	//RGB-->YCrCb
	cvtColor(initialImage, imgYCrCb, cv::COLOR_RGB2YCrCb);

	//PLAN RGB
	//Mat rgbchannel1[3];
	//split(initialImage, rgbchannel1);
	//imshow("Plan R-RGB", rgbchannel1[0]);
	//imshow("Plan G-RGB", rgbchannel1[1]);
	//imshow("Plan B-RGB", rgbchannel1[2]);

	//PLAN GRAYSCALE
	//imshow("Plan Grayscale", imgGRAY);

	//PLAN RGBA
	//Mat rgbchannel2[4];
	//split(imgRGBA, rgbchannel2);
	//imshow("Plan R-RGBA", rgbchannel2[0]);
	//imshow("Plan G-RGBA", rgbchannel2[1]);
	//imshow("Plan B-RGBA", rgbchannel2[2]);
	//imshow("Plan A-RGBA", rgbchannel2[3]);

	//PLAN HSV
	//Mat rgbchannel3[3];
	//split(imgHSV, rgbchannel3);
	//imshow("Plan H-HSV", rgbchannel3[0]);
	//imshow("Plan S-HSV", rgbchannel3[1]);
	//imshow("Plan V-HSV", rgbchannel3[2]);

	//PLAN LAB
	/*Mat rgbchannel4[3];
	split(imgLAB, rgbchannel4);
	imshow("Plan L-LAB", rgbchannel4[0]);
	imshow("Plan A-LAB", rgbchannel4[1]);
	imshow("Plan B-LAB", rgbchannel4[2]);*/

	//PLAN XYZ
	/*Mat rgbchannel5[3];
	split(imgXYZ, rgbchannel5);
	imshow("Plan X-XYZ", rgbchannel5[0]);
	imshow("Plan Y-XYZ", rgbchannel5[1]);
	imshow("Plan Z-XYZ", rgbchannel5[2]);*/

	//PLAN YUV
/*	Mat rgbchannel6[3];
	split(imgYUV, rgbchannel6);
	imshow("Plan Y-YUV", rgbchannel6[0]);
	imshow("Plan U-YUV", rgbchannel6[1]);
	imshow("Plan V-YUV", rgbchannel6[2]);*/

	//PLAN HLS*
/*	Mat rgbchannel7[3];
	split(imgHLS, rgbchannel7);
	imshow("Plan H-HLS", rgbchannel7[0]);
	imshow("Plan L-HLS", rgbchannel7[1]);
	imshow("Plan S-HLS", rgbchannel7[2]);*/

	//PLAN LUV
/*	Mat rgbchannel8[3];
	split(imgLUV, rgbchannel8);
	imshow("Plan L-LUV", rgbchannel8[0]);
	imshow("Plan U-LUV", rgbchannel8[1]);
	imshow("Plan V-LUV", rgbchannel8[2]);*/

	//PLAN YCrCb
	split(imgYCrCb, rgbchannel9);
	/*	imshow("Plan Y-YCrCB", rgbchannel9[0]);
		imshow("Plan Cr-YCrCb", rgbchannel9[1]);
		imshow("Plan Cb-YCrCb", rgbchannel9[2]);*/

		//waitKey(0);
	//}

	return rgbchannel9[0];

}


Mat binarizare(Mat initialImage) {
	//deschidem imaginea color
	char fname[MAX_PATH];
	//Mat initialImage;
	Mat  imgYCrCb;
	Mat  imgYUV;
	Mat  imgXYZ;
	Mat  imgHSV;
	Mat_<uchar> imgBin;
	Mat_<uchar> imgBin2;
	Mat_<uchar> imgBin3;
	Mat_<uchar> imgBin4;
	//while (openFileDlg(fname)) {
		//initialImage = imread(fname, IMREAD_COLOR);

		//YCrCb
	cvtColor(initialImage, imgYCrCb, cv::COLOR_RGB2YCrCb);
	////YUV
	//cvtColor(initialImage, imgYUV, cv::COLOR_RGB2YUV);
	////XYZ
	//cvtColor(initialImage, imgXYZ, cv::COLOR_RGB2XYZ);
	////HSV
	//cvtColor(initialImage, imgHSV, cv::COLOR_RGB2HSV);

	//PLAN YCrCb
	Mat rgbchannel9[3];
	split(imgYCrCb, rgbchannel9);
	imgBin = binarization(rgbchannel9[0]);
	//imshow("Plan Y-YCrCB binarizat", imgBin);

	////PLAN YUV
	//Mat rgbchannel6[3];
	//split(imgYUV, rgbchannel6);
	//imgBin2 = binarization(rgbchannel6[0]);
	//imshow("Plan Y-YUV binarizat", imgBin2);

	////PLAN XYZ
	//Mat rgbchannel5[3];
	//split(imgXYZ, rgbchannel5);
	//imgBin3 = binarization(rgbchannel5[0]);
	//imshow("Plan X-XYZ binarizat", imgBin3);

	////PLAN HSV
	//Mat rgbchannel3[3];
	//split(imgHSV, rgbchannel3);
	//imgBin4 = binarization(rgbchannel3[1]);
	//imshow("Plan S-HSV binarizata", imgBin4);

	//waitKey(0);
//}
	return imgBin;
}


void dilate_erode_methods() {
	char fname[MAX_PATH];
	Mat initialImage;
	Mat_<uchar> img_erode;
	Mat_<uchar> img_dilate;

	int morph_size = 2;
	Mat element = getStructuringElement(
		MORPH_RECT, Size(2 * morph_size + 1,
			2 * morph_size + 1),
		Point(morph_size, morph_size));

	while (openFileDlg(fname)) {
		initialImage = imread(fname, IMREAD_GRAYSCALE);
		erode(initialImage, img_erode, element, Point(-1, -1), 2, 1, 1);
		dilate(initialImage, img_dilate, element, Point(-1, -1), 2, 1, 1);
		imshow("Initial Image", initialImage);
		cvMoveWindow("Initiak Image", 0, 0);
		imshow("Dilated Image", img_dilate);
		cvMoveWindow("Dilated Image", 500, 500);
		imshow("Eroded Image", img_erode);
		cvMoveWindow("Eroded Image", 0, 500);

	}
	waitKey(0);
}


Mat findnegativeImage(Mat image)
{
	Mat new_image;
	//Mat image = imread("Images/imagine2.tiff", IMREAD_COLOR);
	// initialize the output matrix with zeros
	new_image = Mat::zeros(image.size(), image.type());
	// create a matrix with all elements equal to 255 for subtraction
	Mat sub_mat = Mat::ones(image.size(), image.type()) * 255;
	//subtract the original matrix by sub_mat to give the negative output new_image
	subtract(sub_mat, image, new_image);
	// Create Windows
	//namedWindow("Original Image", 1);
	//namedWindow("New Image", 1);
	// Show stuff
	//imshow("Original Image", image);
	//cvMoveWindow("Original Image", 0, 0);
	//imshow("Negative image", new_image);
	//cvMoveWindow("Negative image", 500, 0);
	// Wait until user press some key
	//waitKey(0);
	return new_image;
}

Mat findDrawCountours(Mat image)
{
	//Mat image = imread("Images/imagine4.tiff", IMREAD_COLOR);
	//Prepare the image for findContours
	//cvtColor(image, image, CV_BGR2GRAY);
	//threshold(image, image, 128, 255, CV_THRESH_BINARY);

	//Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
	vector<std::vector<cv::Point> > contours;
	Mat contourOutput = image.clone();
	findContours(contourOutput, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	//Draw the contours
	Mat contourImage(image.size(), CV_8UC3, Scalar(0, 0, 0));
	Scalar colors[3];
	colors[0] = Scalar(255, 0, 255);
	colors[1] = Scalar(0, 255, 255);
	colors[2] = Scalar(255, 255, 153);
	for (size_t idx = 0; idx < contours.size(); idx++) {
		cv::drawContours(contourImage, contours, idx, colors[idx % 3]);
	}

	//cv::imshow("Input Image", image);
	//cvMoveWindow("Input Image", 0, 400);
	//cv::imshow("Contours", contourImage);
	//cvMoveWindow("Contours", 500, 400);
	//cv::waitKey(0);

	return contourImage;
}

vector<std::vector<cv::Point> > findDrawCountoursVector(Mat image)
{
	//Mat image = imread("Images/imagine4.tiff", IMREAD_COLOR);
	//Prepare the image for findContours
	//cvtColor(image, image, CV_BGR2GRAY);
	//threshold(image, image, 128, 255, CV_THRESH_BINARY);

	//Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
	vector<std::vector<cv::Point> > contours;
	Mat contourOutput = image.clone();
	findContours(contourOutput, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	//Draw the contours
	Mat contourImage(image.size(), CV_8UC3, Scalar(0, 0, 0));
	Scalar colors[3];
	colors[0] = Scalar(255, 0, 255);
	colors[1] = Scalar(0, 255, 255);
	colors[2] = Scalar(255, 255, 153);
	for (size_t idx = 0; idx < contours.size(); idx++) {
		cv::drawContours(contourImage, contours, idx, colors[idx % 3]);
	}

	//cv::imshow("Input Image", image);
	//cvMoveWindow("Input Image", 0, 400);
	//cv::imshow("Contours", contourImage);
	//cvMoveWindow("Contours", 500, 400);
	//cv::waitKey(0);

	return contours;
}

void removeSmallObjects() {
	char fname[MAX_PATH];
	Mat initialImage;
	Mat imgYCrCb;
	Mat_<uchar> imgY;
	Mat imgBlur;
	Mat imgThresh;
	vector<vector<Point>> imgCont;
	vector<Vec4i> hierarchy;
	Mat element;
	Mat imgMorphed;

	while (openFileDlg(fname)) {
		initialImage = imread(fname, IMREAD_COLOR);

		cvtColor(initialImage, imgYCrCb, cv::COLOR_RGB2YCrCb);
		Mat rgbchannel9[3];
		split(imgYCrCb, rgbchannel9);
		imgY = rgbchannel9[0];
		imshow("Plan Y", imgY);

		GaussianBlur(imgY, imgBlur, Size(3, 3), 0);
		threshold(imgBlur, imgThresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

		findContours(imgThresh, imgCont, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
		Mat drawing = Mat::zeros(imgThresh.size(), CV_8UC3);
		for (size_t i = 0; i < imgCont.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
			drawContours(drawing, imgCont, (int)i, color, 2, LINE_8, hierarchy, 0);
		}

		element = getStructuringElement(MORPH_RECT, Point(5, 5));

		morphologyEx(imgThresh, imgMorphed, MORPH_CLOSE, element, Point(2));

		imshow("thresh", imgThresh);
		waitKey(0);
	}
}

Mat detectCircles(Mat src) {
	Mat gray;
	Mat dst = src.clone();
	//cvtColor(dst, gray, COLOR_BGR2GRAY);
	medianBlur(dst, dst, 3);
	vector<Vec3f> circles;
	HoughCircles(dst, circles, CV_HOUGH_GRADIENT, 1, dst.rows / 16, 100, 30, 5, 30);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Vec3i c = circles[i];
		Point center = Point(c[0], c[1]);
		// circle center
		circle(dst, center, 1, Scalar(0, 100, 100), 3, LINE_AA);
		// circle outline
		int radius = c[2];
		circle(dst, center, radius, Scalar(255, 0, 255), 3, LINE_AA);
	}

	return dst;
}

vector<float> computeThinnessFactor(vector<std::vector<cv::Point> > contours) {
	vector<float> thinnessVector;
	for (unsigned int i = 0; i < contours.size(); i++)
	{
		//std::cout << "# of contour points: " << contours[i].size() << std::endl;

		for (unsigned int j = 0; j < contours[i].size(); j++)
		{
			//std::cout << "Point(x,y)=" << contours[i][j] << std::endl;
		}
		//std::cout << " Area: " << contourArea(contours[i]) << std::endl;
		//std::cout << " Perimeter: " << arcLength(contours[i], true) << std::endl;
		//std::cout << " Thinness factor " << 4 * PI * contourArea(contours[i]) / (arcLength(contours[i], true) * arcLength(contours[i], true)) << std::endl;
		float var = 4 * PI * contourArea(contours[i]) / (arcLength(contours[i], true) * arcLength(contours[i], true));
		thinnessVector.push_back(var);
	}
	
	return thinnessVector;
}


void maxShape(vector<float> shapes) {
	cout << *max_element(shapes.begin(), shapes.end()) << endl;
}


vector<float> getGoodShapes(vector<float> shapes) {
	vector<float> goodShapes;
	for (auto vec : shapes) {
		if (vec > MAXTHINNESS) {
			goodShapes.push_back(vec);
		}
	}
	return goodShapes;
}


vector<float> getBadShapes(vector<float> shapes) {
	vector<float> badShapes;
	for (auto vec : shapes) {
		if (vec < MAXTHINNESS) {
			badShapes.push_back(vec);
		}
	}
	return badShapes;
}


int main()
{
	Mat initialImage;
	images img;
	Mat new_contour;
	Mat opening;
	Mat closing;
	Mat dil;
	Mat erod;
	Mat circles;
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		initialImage = imread(fname);
		//YCrCb_image = exploreColorSpaces(initialImage);
		//cvtColor(YCrCb_image, grayscale_image, COLOR_BGR2GRAY);   //Converting BGR to Grayscale image and storing it into converted matrix//
		//threshold(YCrCb_image, img_bw, 100, 255, THRESH_BINARY);
		img.img_bw = binarizare(initialImage);
		img.negative_image = findnegativeImage(img.img_bw);

		int morph_size = 2;
		Mat element = getStructuringElement(
			MORPH_RECT, Size(2 * morph_size + 1,
				2 * morph_size + 1),
			Point(morph_size, morph_size));
		Mat erod, dill;

		// For Erosion
		erode(img.img_bw, erod, element,
			Point(-1, -1), 1);

		// For Dilation
		dilate(img.img_bw, dil, element,
			Point(-1, -1), 1);

		// For Opening
		morphologyEx(erod, opening,
			MORPH_OPEN, element,
			Point(-1, -1), 2);

		img.contour_image = findDrawCountours(opening);
		new_contour = findDrawCountours(opening);

		//circles = detectCircles(opening);

		vector<std::vector<cv::Point> > contours = findDrawCountoursVector(opening);

	    /*int i = 0;
		for (auto vec : contours) {
			std::cout << "Object" << i << "\n";
			for (auto v : vec)
				std::cout << v << std::endl;
			i++;
		}*/

		vector<float> thinnessVector = computeThinnessFactor(contours);
		//for (auto v : thinnessVector)
			//std::cout << v << " " << std::endl;
		//maxShape(thinnessVector);

		//PRINTAM GOOD SHAPES
		vector<float> goodShapes;
		goodShapes = getGoodShapes(thinnessVector);
		cout << "Good shapes" << std::endl;
		for (auto v : goodShapes)
			std::cout << v << " " << std::endl;

		//PRINTAM Bad SHAPES
		vector<float> badShapes;
		badShapes = getBadShapes(thinnessVector);
		cout << "Bad shapes" << std::endl;
		for (auto v : badShapes)
			std::cout << v << " " << std::endl;

		//imshow("YCrCb image", YCrCb_image);
		imshow("Original Image", initialImage);
		imshow("Binarized image", img.img_bw);
		imshow("Negative image", img.negative_image);
		imshow("Dilate", dil); //nu merge
		imshow("Erode", erod);
		imshow("Open", opening);
		imshow("Countours", new_contour);  //alea ce ne intereseaza sunt galbene
		//imshow("Circles", circles);

		waitKey(0);
	}

	return 0;
}