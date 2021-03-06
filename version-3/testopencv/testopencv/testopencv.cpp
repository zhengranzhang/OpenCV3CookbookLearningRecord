// testopencv.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include <sstream>

#include <string>
#include <unordered_set>
#include <random>


#include <opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/bgsegm.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/face.hpp>

//#include "robustMatcher.h"
//#include "targetMatcher.h"
//#include "CameraCalibrator.h"
//#include "videoprocessor.h"
//#include "BGFGSegmentor.h"
//#include "featuretracker.h"
//#include "visualTracker.h"

//// triangulate using Linear LS-Method
//cv::Vec3d triangulate(const cv::Mat& p1, const cv::Mat& p2, const cv::Vec2d& u1, const cv::Vec2d& u2)
//{
//	// system of equations assuming image=[u,v] and X=[x,y,z,1] from u(p3.X)= p1.X and v(p3.X)=p2.X
//	cv::Matx43d A(u1(0)*p1.at<double>(2, 0) - p1.at<double>(0, 0), 
//				  u1(0)*p1.at<double>(2, 1) - p1.at<double>(0, 1), 
//		          u1(0)*p1.at<double>(2, 2) - p1.at<double>(0, 2),
//				  u1(1)*p1.at<double>(2, 0) - p1.at<double>(1, 0),
//		          u1(1)*p1.at<double>(2, 1) - p1.at<double>(1, 1), 
//		          u1(1)*p1.at<double>(2, 2) - p1.at<double>(1, 2),
//		          u2(0)*p2.at<double>(2, 0) - p2.at<double>(0, 0), 
//		          u2(0)*p2.at<double>(2, 1) - p2.at<double>(0, 1), 
//		          u2(0)*p2.at<double>(2, 2) - p2.at<double>(0, 2),
//		          u2(1)*p2.at<double>(2, 0) - p2.at<double>(1, 0), 
//		          u2(1)*p2.at<double>(2, 1) - p2.at<double>(1, 1), 
//		          u2(1)*p2.at<double>(2, 2) - p2.at<double>(1, 2));
//
//	cv::Matx41d B(p1.at<double>(0, 3) - u1(0)*p1.at<double>(2, 3),
//		          p1.at<double>(1, 3) - u1(1)*p1.at<double>(2, 3),
//		          p2.at<double>(0, 3) - u2(0)*p2.at<double>(2, 3),
//		          p2.at<double>(1, 3) - u2(1)*p2.at<double>(2, 3));
//	// X contains the 3D coordinate of the reconstructed point
//	cv::Vec3d X;
//	// solve AX=B
//	cv::solve(A, B, X, cv::DECOMP_SVD);
//	return X;
//}

//void triangulate(const cv::Mat &p1, const cv::Mat &p2, const std::vector<cv::Vec2d> &pts1, const std::vector<cv::Vec2d> &pts2, std::vector<cv::Vec3d> &pts3D)
//{
//	for (int i = 0; i < pts1.size(); i++)
//	{
//		pts3D.push_back(triangulate(p1,p2,pts1[i],pts2[i]));
//	}
//}

//void draw(const cv::Mat& img, cv::Mat& out)
//{
//	img.copyTo(out);
//	cv::circle(out, cv::Point(100,100), 5, cv::Scalar(255,0,0),2);
//}

//void canny(cv::Mat& img, cv::Mat& out)
//{
//	// Convert to gray
//	if (img.channels() == 3)
//	{
//		cv::cvtColor(img, out, cv::COLOR_BGR2GRAY);
//	}
//	// Compute Canny edges
//	cv::Canny(out, out, 100, 200);
//	// Invert the image
//	cv::threshold(out, out, 128, 255, cv::THRESH_BINARY_INV);
//}

//void drawOpticalFlow(const cv::Mat& oflow, cv::Mat& flowImage, int stride, float scale, const cv::Scalar& color)
//{
//	// create the image if required
//	if (flowImage.size() != oflow.size())
//	{
//		flowImage.create(oflow.size(), CV_8UC3);
//		flowImage = cv::Vec3i(255,255,255);
//	}
//	// for all vectors using stride as a step
//	for (int y = 0; y < oflow.rows; y += stride)
//	{
//		for (int x = 0; x < oflow.cols; x += stride)
//		{
//			cv::Point2f vector = oflow.at<cv::Point2f>(y, x);
//			cv::line(flowImage, cv::Point(x,y), cv::Point(static_cast<int>(x + scale*vector.x + 0.5), static_cast<int>(y + scale * vector.y + 0.5)), color);
//			cv::circle(flowImage, cv::Point(static_cast<int>(x + scale * vector.x + 0.5), static_cast<int>(y + scale * vector.y + 0.5)), 1, color, -1);
//		}
//	}
//
//}

// compute the Local Binary Patterns of a gray-level image
//void lbp(const cv::Mat& image, cv::Mat& result)
//{
//	assert(image.channels() == 1); // input image must be gray scale
//	result.create(image.size(), CV_8U);
//	for (int j = 1; j < image.rows - 1; j++)
//	{
//		// for all rows (except first and last)
//		const uchar* previous = image.ptr<const uchar>(j - 1); // previous row
//		const uchar* current  = image.ptr<const uchar>(j);
//		const uchar* next     = image.ptr<const uchar>(j + 1); // next row
//		uchar* output = result.ptr<uchar>(j); // output row
//		for (int i = 1; i < image.cols - 1; i++)
//		{
//			// compose local binary pattern
//			*output = previous[i - 1] > current[i] ? 1 : 0;
//			*output |= previous[i] > current[i] ? 2 : 0;
//			*output |= previous[i + 1] > current[i] ? 4 : 0;
//
//			*output |= current[i - 1] > current[i] ? 8 : 0;
//			*output |= current[i + 1] > current[i] ? 16 : 0;
//
//			*output |= next[i - 1] > current[i] ? 32 : 0;
//			*output |= next[i] > current[i] ? 64 : 0;
//			*output |= next[i + 1] > current[i] ? 128 : 0;
//
//			output++;
//		}
//	}
//	// Set the unprocess pixels to 0
//	result.row(0).setTo(cv::Scalar(0));
//	result.row(result.rows - 1).setTo(cv::Scalar(0));
//	result.col(0).setTo(cv::Scalar(0));
//	result.col(result.cols - 1).setTo(cv::Scalar(0));
//}

void drawHOG(std::vector<float>::const_iterator hog, int numberOfBins, cv::Mat image, float scale = 1.0)
{
	const float PI = 3.1415927;
	float binStep = PI / numberOfBins;
	float maxLength = image.rows;
	float cx = image.cols / 2.;
	float cy = image.rows / 2.;
	for (int bin = 0; bin < numberOfBins; bin++)
	{
		// bin orientation
		float angle = bin * binStep;
		float dirX = cos(angle);
		float dirY = sin(angle);
		// length of line proportion to bin size
		float length = 0.5 * maxLength ** (hog + bin);
		// drawing the line
		float x1 = cx - dirX * length * scale;
		float y1 = cy - dirY * length * scale;
		float x2 = cx + dirX * length * scale;
		float y2 = cy + dirY * length * scale;
		cv::line(image, cv::Point(x1,y1), cv::Point(x2,y2), CV_RGB(255, 255, 255), 1);
	}

}


void drawHOGDescriptors(const cv::Mat& image, cv::Mat& hogImage, cv::Size cellSize, int nBins)
{
	// block size is image size
	cv::HOGDescriptor hog(cv::Size((image.cols/cellSize.width)*cellSize.width, (image.rows / cellSize.height) * cellSize.height), cv::Size((image.cols / cellSize.width)*cellSize.width, (image.rows / cellSize.height) * cellSize.height), cellSize, cellSize, nBins);
	// compute HOG
	std::vector<float> descriptors;
	hog.compute(image, descriptors);
	float scale = 2.0 / *std::max_element(descriptors.begin(), descriptors.end());
	hogImage.create(image.rows, image.cols, CV_8U);
	std::vector<float>::const_iterator itDesc = descriptors.begin();
	for (int i = 0; i < image.rows / cellSize.height; i++)
	{
		for (int j = 0; j < image.cols / cellSize.width; j++)
		{
			// draw wach cell
			hogImage(cv::Rect(j*cellSize.width, i*cellSize.height, cellSize.width, cellSize.height));
			drawHOG(itDesc, nBins, hogImage(cv::Rect(j*cellSize.width, i*cellSize.height,
				cellSize.width, cellSize.height)), scale);
			itDesc += nBins;
		}
	}
}

int main()
{
	/*cv::Mat image = cv::imread("church01.jpg");
	if (image.empty())
	{
		return 0;
	}
	cv::transpose(image,image);
	cv::flip(image,image,0);
	std::vector<cv::KeyPoint> keypoints;
	keypoints.clear();
	cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> ptrSURF = cv::xfeatures2d::SurfFeatureDetector::create(2000.0);
	ptrSURF->detect(image,keypoints);
	cv::Mat featureImage;
	cv::drawKeypoints(image, keypoints, featureImage, cv::Scalar(255,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("SURF");
	cv::imshow("SURF", featureImage);
	std::cout << "Number of SURF keypoints: " << keypoints.size() << std::endl;
	image = cv::imread("church03.jpg", cv::IMREAD_GRAYSCALE);
	cv::transpose(image,image);
	cv::flip(image,image,0);
	ptrSURF->detect(image, keypoints);
	cv::drawKeypoints(image, keypoints, featureImage, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("SURF (2)");
	cv::imshow("SURF (2)", featureImage);
	image = cv::imread("church01.jpg", cv::IMREAD_GRAYSCALE);
	cv::transpose(image,image);
	cv::flip(image,image,0);
	keypoints.clear();
	cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> ptrSIFT = cv::xfeatures2d::SiftFeatureDetector::create();
	ptrSIFT->detect(image, keypoints);
	cv::drawKeypoints(image, keypoints, featureImage, cv::Scalar(255,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("SIFT");
	cv::imshow("SIFT", featureImage);
	std::cout << "Number of SIFT keypoints: " << keypoints.size() << std::endl;
	image = cv::imread("church01.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cv::transpose(image, image);
	cv::flip(image, image, 0);
	keypoints.clear();
	cv::Ptr<cv::BRISK> ptrBRISK = cv::BRISK::create(60,5);
	ptrBRISK->detect(image,keypoints);
	cv::drawKeypoints(image, keypoints, featureImage, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("BRISK");
	cv::imshow("BRISK", featureImage);
	image = cv::imread("church01.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cv::transpose(image, image);
	cv::flip(image, image, 0);
	keypoints.clear();
	cv::Ptr<cv::ORB> ptrORB = cv::ORB::create(75,1.2,8);
	ptrORB->detect(image, keypoints);
	cv::drawKeypoints(image, keypoints, featureImage, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("ORB");
	cv::imshow("ORB", featureImage);*/

	/*cv::Mat image1 = cv::imread("church01.jpg",cv::IMREAD_GRAYSCALE);
	cv::Mat image2 = cv::imread("church02.jpg",cv::IMREAD_GRAYSCALE);
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;
	cv::Ptr<cv::Feature2D> ptrFeature2D = cv::xfeatures2d::SURF::create(2000.0);
	ptrFeature2D->detect(image1,keypoints1);
	ptrFeature2D->detect(image2, keypoints2);
	cv::Mat featureImage;
	cv::drawKeypoints(image1, keypoints1, featureImage, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("SURF");
	cv::imshow("SURF", featureImage);
	cv::Mat descriptors1;
	cv::Mat descriptors2;
	ptrFeature2D->compute(image1,keypoints1,descriptors1);
	ptrFeature2D->compute(image2,keypoints2, descriptors2);
	cv::BFMatcher matcher(cv::NORM_L2);
	std::vector<cv::DMatch> matches;
	matcher.match(descriptors1,descriptors2,matches);
	cv::Mat imageMatches;
	cv::drawMatches(image1,keypoints1,image2,keypoints2,matches,imageMatches,cv::Scalar(255,255,255),cv::Scalar(255,255,255),std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS|cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("SURF Matches");
	cv::imshow("SURF Matches", imageMatches);
	std::cout << "Number of matches: " << matches.size() << std::endl;

	std::vector<std::vector<cv::DMatch>> matches2;
	matcher.knnMatch(descriptors1,descriptors2,matches2,2);
	matches.clear();
	double ratioMax = 0.6;
	std::vector<std::vector<cv::DMatch>>::iterator it;
	for (it = matches2.begin(); it != matches2.end(); it++)
	{
		if ((*it)[0].distance / (*it)[1].distance < ratioMax)
		{
			matches.push_back((*it)[0]);
		}
	}
	cv::drawMatches(image1,keypoints1,image2,keypoints2,matches,imageMatches, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS|cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	std::cout << "Number of matches (after ratio test): " << matches.size() << std::endl;
	cv::namedWindow("SURF Matches (ratio test at 0.6)");
	cv::imshow("SURF Matches (ratio test at 0.6)", imageMatches);
	float maxDist = 0.3;
	matches2.clear();
	matcher.radiusMatch(descriptors1,descriptors2,matches2,maxDist);
	cv::drawMatches(image1,keypoints1,image2,keypoints2,matches2,imageMatches,cv::Scalar(255,255,255),cv::Scalar(255, 255, 255),std::vector<std::vector<char>>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS|cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	int nmatches = 0;
	for (int i = 0; i < matches2.size(); i++)
		nmatches += matches2[i].size();
	std::cout << "Number of matches (with max radius): " << nmatches << std::endl;
	cv::namedWindow("SURF Matches (with max radius)");
	cv::imshow("SURF Matches (with max radius)", imageMatches);
	image1 = cv::imread("church01.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	image2 = cv::imread("church03.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	std::cout << "Number of SIFT keypoints (image 1): " << keypoints1.size() << std::endl;
	std::cout << "Number of SIFT keypoints (image 2): " << keypoints2.size() << std::endl;
	ptrFeature2D = cv::xfeatures2d::SIFT::create();
	ptrFeature2D->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
	ptrFeature2D->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
	matcher.match(descriptors1,descriptors2,matches);
	std::nth_element(matches.begin(),matches.begin()+50,matches.end());
	matches.erase(matches.begin()+50,matches.end());
	cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, imageMatches, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("Multi-scale SIFT Matches");
	cv::imshow("Multi-scale SIFT Matches", imageMatches);*/

	//cv::Mat image1 = cv::imread("church01.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//cv::Mat image2 = cv::imread("church02.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//std::vector<cv::KeyPoint> keypoints1;
	//std::vector<cv::KeyPoint> keypoints2;
	//cv::Mat descriptors1;
	//cv::Mat descriptors2;
	//cv::Ptr<cv::Feature2D> feature = cv::ORB::create(60); 
	////cv::Ptr<cv::Feature2D> feature = cv::BRISK::create(80);
	//feature->detectAndCompute(image1,cv::noArray(),keypoints1,descriptors1);
	//feature->detectAndCompute(image2,cv::noArray(),keypoints2,descriptors2);
	//cv::Mat featureImage;
	//cv::drawKeypoints(image1,keypoints1,featureImage,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//cv::namedWindow("ORB");
	//cv::imshow("ORB", featureImage);
	//feature = cv::xfeatures2d::FREAK::create();
	//feature->compute(image1, keypoints1, descriptors1);
	//feature->compute(image2, keypoints2, descriptors2);
	//cv::BFMatcher matcher(cv::NORM_HAMMING);
	//std::vector<cv::DMatch> matches;
	//matcher.match(descriptors1,descriptors2,matches);
	//cv::Mat imageMatches;
	//cv::drawMatches(image1,keypoints1,image2,keypoints2,matches,imageMatches, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS|cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//cv::namedWindow("ORB Matches");
	//cv::imshow("ORB Matches", imageMatches);
	////cv::namedWindow("FREAK Matches");
	////cv::imshow("FREAK Matches", imageMatches);

	//cv::Mat image1 = cv::imread("church01.jpg", 0);
	//cv::Mat image2 = cv::imread("church02.jpg", 0);
	//if (!image1.data || !image2.data)
	//	return 0;
	///*cv::namedWindow("Right Image");
	//cv::imshow("Right Image", image1);
	//cv::namedWindow("Left Image");
	//cv::imshow("Left Image", image2);*/
	//std::vector<cv::KeyPoint> keypoints1;
	//std::vector<cv::KeyPoint> keypoints2;
	//cv::Mat descriptors1, descriptors2;
	//cv::Ptr<cv::Feature2D> ptrFeature2D = cv::xfeatures2d::SIFT::create(74);
	//ptrFeature2D->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
	//ptrFeature2D->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
	////cv::Mat imageKP;
	////cv::drawKeypoints(image1,keypoints1,imageKP,cv::Scalar(255,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	///*cv::namedWindow("Right SIFT Features");
	//cv::imshow("Right SIFT Features", imageKP);*/
	////cv::drawKeypoints(image2, keypoints2, imageKP, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	///*cv::namedWindow("Left SIFT Features");
	//cv::imshow("Left SIFT Features", imageKP);*/
	//cv::BFMatcher matcher(cv::NORM_L2,true);
	//std::vector<cv::DMatch> matches;
	//matcher.match(descriptors1,descriptors2,matches);
	//std::vector<cv::DMatch> selMatches;
	//selMatches.push_back(matches[2]);
	//selMatches.push_back(matches[5]);
	//selMatches.push_back(matches[16]);
	//selMatches.push_back(matches[19]);
	//selMatches.push_back(matches[14]);
	//selMatches.push_back(matches[34]);
	//selMatches.push_back(matches[29]);
	//cv::Mat imageMatches;
	//cv::drawMatches(image1,keypoints1,image2,keypoints2,selMatches,imageMatches,cv::Scalar(255,255,255), cv::Scalar(255, 255, 255),std::vector<char>(),2);
	//cv::namedWindow("Matches");
	//cv::imshow("Matches", imageMatches);
	//std::vector<int> pointIndexes1;
	//std::vector<int> pointIndexes2;
	//for (std::vector<cv::DMatch>::const_iterator it = selMatches.begin(); it != selMatches.end(); ++it)
	//{
	//	pointIndexes1.push_back(it->queryIdx);
	//	pointIndexes2.push_back(it->trainIdx);
	//}
	//std::vector<cv::Point2f> selPoints1, selPoints2;
	//cv::KeyPoint::convert(keypoints1,selPoints1,pointIndexes1);
	//cv::KeyPoint::convert(keypoints2, selPoints2, pointIndexes2);
	//std::vector<cv::Point2f>::const_iterator it = selPoints1.begin();
	//while (it != selPoints1.end()) {
	//	// draw a circle at each corner location
	//	cv::circle(image1, *it, 3, cv::Scalar(255, 255, 255), 2);
	//	++it;
	//}
	//it = selPoints2.begin();
	//while (it != selPoints2.end()) {
	//	// draw a circle at each corner location
	//	cv::circle(image2, *it, 3, cv::Scalar(255, 255, 255), 2);
	//	++it;
	//}
	////cv::namedWindow("Right Image");
	////cv::imshow("Right Image", image1);
	////cv::namedWindow("Left Image");
	////cv::imshow("Left Image", image2);
	//cv::Mat fundamental = cv::findFundamentalMat(selPoints1,selPoints2,cv::FM_7POINT);
	//std::cout << "F-Matrix size= " << fundamental.rows << "," << fundamental.cols << std::endl;
	//cv::Mat fund(fundamental, cv::Rect(0, 0, 3, 3));
	//std::cout << "size of F matrix:" << fund.rows << "x" << fund.cols << std::endl;
	//std::vector<cv::Vec3f> lines1;
	//cv::computeCorrespondEpilines(selPoints1,1,fund,lines1);
	//for (std::vector<cv::Vec3f>::const_iterator it = lines1.begin(); it != lines1.end(); ++it)
	//{
	//	cv::line(image2,cv::Point(0,-(*it)[2]/(*it)[1]), cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image1.cols) / (*it)[1]), cv::Scalar(255, 255, 255));
	//}
	//std::vector<cv::Vec3f> lines2;
	//cv::computeCorrespondEpilines(cv::Mat(selPoints2), 2, fund, lines2);
	//for (std::vector<cv::Vec3f>::const_iterator it = lines2.begin(); it != lines2.end(); ++it)
	//{
	//	cv::line(image1, cv::Point(0, -(*it)[2] / (*it)[1]), cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image1.cols) / (*it)[1]), cv::Scalar(255, 255, 255));
	//}
	//cv::Mat both(image1.rows, image1.cols + image2.cols, CV_8U);
	//image1.copyTo(both.colRange(0, image1.cols));
	//image2.copyTo(both.colRange(image1.cols, image1.cols + image2.cols));
	//cv::namedWindow("Epilines");
	//cv::imshow("Epilines", both);

	////std::vector<cv::Point2f> points1, points2, newPoints1, newPoints2;
	////cv::KeyPoint::convert(keypoints1, points1);
	////cv::KeyPoint::convert(keypoints2, points2);
	////cv::correctMatches(fund, points1, points2, newPoints1, newPoints2);
	////cv::KeyPoint::convert(newPoints1, keypoints1);
	////cv::KeyPoint::convert(newPoints2, keypoints2);
	////cv::drawMatches(image1,keypoints1,image2,keypoints2,matches,imageMatches, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), std::vector<char>(),2 );
	////cv::namedWindow("Corrected matches");
	////cv::imshow("Corrected matches", imageMatches);

	//cv::Mat image1 = cv::imread("church01.jpg", 0);
	//cv::Mat image2 = cv::imread("church03.jpg", 0);
	//if (!image1.data || !image2.data)
	//	return 0;
	//RobustMatcher rmatcher(cv::xfeatures2d::SIFT::create(250));
	//std::vector<cv::DMatch> matches;
	//std::vector<cv::KeyPoint> keypoints1, keypoints2;
	//cv::Mat fundamental = rmatcher.match(image1,image2,matches,keypoints1,keypoints2);
	//cv::Mat imageMatches;
	//cv::drawMatches(image1,keypoints1,image2,keypoints2,matches,imageMatches, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), std::vector<char>(),2);
	//cv::namedWindow("Matches");
	//cv::imshow("Matches", imageMatches);

	//cv::Mat image1 = cv::imread("parliament1.jpg", 0);
	//cv::Mat image2 = cv::imread("parliament2.jpg", 0);
	//if (!image1.data || !image2.data)
	//	return 0;
	//std::vector<cv::KeyPoint> keypoints1;
	//std::vector<cv::KeyPoint> keypoints2;
	//cv::Mat descriptors1, descriptors2;
	//cv::Ptr<cv::Feature2D> ptrFeature2D = cv::xfeatures2d::SIFT::create(74);
	//ptrFeature2D->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
	//ptrFeature2D->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
	//cv::BFMatcher matcher(cv::NORM_L2,true);
	//std::vector<cv::DMatch> matches;
	//matcher.match(descriptors1,descriptors2,matches);
	//cv::Mat imageMatches;
	//cv::drawMatches(image1,keypoints1,image2,keypoints2,matches,imageMatches,cv::Scalar(255,255,255), cv::Scalar(255, 255, 255),std::vector<char>(),2);
	//cv::namedWindow("Matches (pure rotation case)");
	//cv::imshow("Matches (pure rotation case)", imageMatches);
	//std::vector<cv::Point2f> points1, points2;
	//for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
	//{
	//	// Get the position of left keypoints
	//	float x = keypoints1[it->queryIdx].pt.x;
	//	float y = keypoints1[it->queryIdx].pt.y;
	//	points1.push_back(cv::Point2f(x,y));
	//	// Get the position of right keypoints
	//	x = keypoints2[it->trainIdx].pt.x;
	//	y = keypoints2[it->trainIdx].pt.y;
	//	points2.push_back(cv::Point2f(x,y));
	//}
	//// Find the homography between image 1 and image 2
	//std::vector<char> inliers;
	//cv::Mat homography = cv::findHomography(points1,points2,inliers,cv::RANSAC,1.0);
	//cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, imageMatches, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), inliers, 2);
	//cv::namedWindow("Homography inlier points");
	//cv::imshow("Homography inlier points", imageMatches);
	//cv::Mat result;
	//cv::warpPerspective(image1,result,homography,cv::Size(2*image1.cols,image1.rows));
	//cv::Mat half(result, cv::Rect(0, 0, image2.cols, image2.rows));
	//image2.copyTo(half);
	//cv::namedWindow("Image mosaic");
	//cv::imshow("Image mosaic", result);
	//std::vector<cv::Mat> images;
	//images.push_back(cv::imread("parliament1.jpg"));
	//images.push_back(cv::imread("parliament2.jpg"));
	//cv::Mat panorama;
	//cv::Stitcher stitcher = cv::Stitcher::createDefault();
	//cv::Stitcher::Status status = stitcher.stitch(images, panorama);
	//if (status == cv::Stitcher::OK)
	//{
	//	// Display the panorama
	//	cv::namedWindow("Panorama");
	//	cv::imshow("Panorama", panorama);
	//}

	//cv::Mat target = cv::imread("cookbook1.bmp",0);
	//cv::Mat image = cv::imread("objects.jpg", 0);
	//if (!target.data || !image.data)
	//	return 0;
	////cv::namedWindow("Target");
	////cv::imshow("Target", target);
	////cv::namedWindow("Image");
	////cv::imshow("Image", image);
	//TargetMatcher tmatcher(cv::FastFeatureDetector::create(10), cv::BRISK::create());
	//tmatcher.setNormType(cv::NORM_HAMMING);
	//std::vector<cv::DMatch> matches;
	//std::vector<cv::KeyPoint> keypoints1, keypoints2;
	//std::vector<cv::Point2f> corners;
	//tmatcher.setTarget(target);
	//tmatcher.detectTarget(image, corners);
	//// draw the target corners on the image
	//if (corners.size() == 4)
	//{
	//	cv::line(image, cv::Point(corners[0]), cv::Point(corners[1]), cv::Scalar(255, 255, 255), 3);
	//	cv::line(image, cv::Point(corners[1]), cv::Point(corners[2]), cv::Scalar(255, 255, 255), 3);
	//	cv::line(image, cv::Point(corners[2]), cv::Point(corners[3]), cv::Scalar(255, 255, 255), 3);
	//	cv::line(image, cv::Point(corners[3]), cv::Point(corners[0]), cv::Scalar(255, 255, 255), 3);
	//}
	//cv::namedWindow("Target detection");
	//cv::imshow("Target detection", image);

	//cv::Mat image;
	//std::vector<std::string> filelist;
	//// generate list of chessboard image filename
	//for (int i = 1; i <= 27; i++)
	//{
	//	std::stringstream str;
	//	str << "chessboards/chessboard" << std::setw(2) << std::setfill('0') << i << ".jpg";
	//	std::cout << str.str() << std::endl;
	//	filelist.push_back(str.str());
	//	image = cv::imread(str.str(),0);
	//	cv::imshow("Board Image",image);
	//	cv::waitKey(100);
	//}
	//// Create calibrator object
	//CameraCalibrator cameraCalibrator;
	//// add the corners from the chessboard
	//cv::Size boardSize(7,5);
	//cameraCalibrator.addChessboardPoints(filelist,boardSize,"Detected points");
	//cameraCalibrator.setCalibrationFlag(true,true);
	//cameraCalibrator.calibrate(image.size());
	//image = cv::imread(filelist[14], 0);
	//cv::Size newSize(static_cast<int>(image.cols*1.5), static_cast<int>(image.rows*1.5));
	//cv::Mat uImage = cameraCalibrator.remap(image, newSize);
	//cv::Mat cameraMatrix = cameraCalibrator.getCameraMatrix();
	//std::cout << " Camera intrinsic: " << cameraMatrix.rows << "x" << cameraMatrix.cols << std::endl;
	//std::cout << cameraMatrix.at<double>(0, 0) << " " << cameraMatrix.at<double>(0, 1) << " " << cameraMatrix.at<double>(0, 2) << std::endl;
	//std::cout << cameraMatrix.at<double>(1, 0) << " " << cameraMatrix.at<double>(1, 1) << " " << cameraMatrix.at<double>(1, 2) << std::endl;
	//std::cout << cameraMatrix.at<double>(2, 0) << " " << cameraMatrix.at<double>(2, 1) << " " << cameraMatrix.at<double>(2, 2) << std::endl;
	//cv::namedWindow("Original Image");
	//cv::imshow("Original Image", image);
	//cv::namedWindow("Undistorted Image");
	//cv::imshow("Undistorted Image", uImage);
	//// Store everything in a xml file
	//cv::FileStorage fs("calib.xml", cv::FileStorage::WRITE);
	//fs<< "Intrinsic" << cameraMatrix;
	//fs << "Distortion" << cameraCalibrator.getDistCoeffs();

	//cv::Mat cameraMatrix;
	//cv::Mat cameraDistCoeffs;
	//cv::FileStorage fs("calib.xml", cv::FileStorage::READ);
	//fs["Intrinsic"] >> cameraMatrix;
	//fs["Distortion"] >> cameraDistCoeffs;
	//std::cout << " Camera intrinsic: " << cameraMatrix.rows << "x" << cameraMatrix.cols << std::endl;
	//std::cout << cameraMatrix.at<double>(0, 0) << " " << cameraMatrix.at<double>(0, 1) << " " << cameraMatrix.at<double>(0, 2) << std::endl;
	//std::cout << cameraMatrix.at<double>(1, 0) << " " << cameraMatrix.at<double>(1, 1) << " " << cameraMatrix.at<double>(1, 2) << std::endl;
	//std::cout << cameraMatrix.at<double>(2, 0) << " " << cameraMatrix.at<double>(2, 1) << " " << cameraMatrix.at<double>(2, 2) << std::endl << std::endl;
	//cv::Matx33d cMatrix(cameraMatrix);
	//// Input image points
	//std::vector<cv::Point2f> imagePoints;
	//imagePoints.push_back(cv::Point2f(136, 113));
	//imagePoints.push_back(cv::Point2f(379, 114));
	//imagePoints.push_back(cv::Point2f(379, 150));
	//imagePoints.push_back(cv::Point2f(138, 135));
	//imagePoints.push_back(cv::Point2f(143, 146));
	//imagePoints.push_back(cv::Point2f(381, 166));
	//imagePoints.push_back(cv::Point2f(345, 194));
	//imagePoints.push_back(cv::Point2f(103, 161));
	//// Input object points
	//std::vector<cv::Point3f> objectPoints;
	//objectPoints.push_back(cv::Point3f(0, 45, 0));
	//objectPoints.push_back(cv::Point3f(242.5, 45, 0));
	//objectPoints.push_back(cv::Point3f(242.5, 21, 0));
	//objectPoints.push_back(cv::Point3f(0, 21, 0));
	//objectPoints.push_back(cv::Point3f(0, 9, -9));
	//objectPoints.push_back(cv::Point3f(242.5, 9, -9));
	//objectPoints.push_back(cv::Point3f(242.5, 9, 44.5));
	//objectPoints.push_back(cv::Point3f(0, 9, 44.5));
	//cv::Mat image = cv::imread("bench2.jpg");
	//// Draw image points
	//for (int i = 0; i < 8; i++)
	//{
	//	cv::circle(image,imagePoints[i],3,cv::Scalar(0,0,0),2);
	//}
	//cv::imshow("An image of a bench", image);
	//// Create a viz window
	//cv::viz::Viz3d visualizer("Viz window");
	//visualizer.setBackgroundColor(cv::viz::Color::white());
	//// Construct the scene
	//// Create a virtual camera
	//cv::viz::WCameraPosition cam(cMatrix,image,30.0,cv::viz::Color::black());
	//// Create a virtual bench from cuboids
	//cv::viz::WCube plane1(cv::Point3f(0.0, 45.0, 0.0), cv::Point3f(242.5, 21.0, -9.0), true, cv::viz::Color::blue());
	//plane1.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
	//cv::viz::WCube plane2(cv::Point3f(0.0, 9.0, -9.0), cv::Point3f(242.5, 0.0, 44.5), true, cv::viz::Color::blue());
	//plane2.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
	//// Add the virtual objects to the environment
	//visualizer.showWidget("top", plane1);
	//visualizer.showWidget("bottom", plane2);
	//visualizer.showWidget("Camera", cam);
	//// Get the camera pose from 3D/2D points
	//cv::Mat rvec, tvec;
	//cv::solvePnP(objectPoints,imagePoints,cameraMatrix,cameraDistCoeffs,rvec,tvec);
	//std::cout << " rvec: " << rvec.rows << "x" << rvec.cols << std::endl;
	//std::cout << " tvec: " << tvec.rows << "x" << tvec.cols << std::endl;
	//// convert vector-3 rotation to a 3x3 rotation matrix
	//cv::Mat rotation;
	//cv::Rodrigues(rvec,rotation);
	//// Move the bench
	//cv::Affine3d pose(rotation,tvec);
	//visualizer.setWidgetPose("top", pose);
	//visualizer.setWidgetPose("bottom", pose);
	//// visualization loop
	////while (cv::waitKey(100)==-1 && !visualizer.wasStopped())
	//while (!visualizer.wasStopped())
	//{
	//	visualizer.spinOnce(1,true);
	//}
	//
	//std::cout << "end" << std::endl;

	//// Read input images
	//cv::Mat image1 = cv::imread("soup1.jpg",0);
	//cv::Mat image2 = cv::imread("soup2.jpg",0);
	//if (!image1.data || !image2.data)
	//	return 0;
	//// Display the images
	///*cv::namedWindow("Right Image");
	//cv::imshow("Right Image", image1);
	//cv::namedWindow("Left Image");
	//cv::imshow("Left Image", image2);*/
	//// Read the camera calibration parameters
	//cv::Mat cameraMatrix;
	//cv::Mat cameraDistCoeffs;
	//cv::FileStorage fs("calib.xml", cv::FileStorage::READ);
	//fs["Intrinsic"] >> cameraMatrix;
	//fs["Distortion"] >> cameraDistCoeffs;
	//cameraMatrix.at<double>(0, 2) = 268.;
	//cameraMatrix.at<double>(1, 2) = 178;
	//std::cout << " Camera intrinsic: " << cameraMatrix.rows << "x" << cameraMatrix.cols << std::endl;
	//std::cout << cameraMatrix.at<double>(0, 0) << " " << cameraMatrix.at<double>(0, 1) << " " << cameraMatrix.at<double>(0, 2) << std::endl;
	//std::cout << cameraMatrix.at<double>(1, 0) << " " << cameraMatrix.at<double>(1, 1) << " " << cameraMatrix.at<double>(1, 2) << std::endl;
	//std::cout << cameraMatrix.at<double>(2, 0) << " " << cameraMatrix.at<double>(2, 1) << " " << cameraMatrix.at<double>(2, 2) << std::endl << std::endl;
	//cv::Matx33f cMatrix(cameraMatrix);
	//// vector of keypoints and descriptors
	//std::vector<cv::KeyPoint> keypoints1;
	//std::vector<cv::KeyPoint> keypoints2;
	//cv::Mat descriptors1, descriptors2;
	//// Construction of the SIFT feature detector 
	//cv::Ptr<cv::Feature2D> ptrFeature2D = cv::xfeatures2d::SIFT::create(500);
	//ptrFeature2D->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
	//ptrFeature2D->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
	//std::cout << "Number of feature points (1): " << keypoints1.size() << std::endl;
	//std::cout << "Number of feature points (2): " << keypoints2.size() << std::endl;
	//// Match the two image descriptors
	//cv::BFMatcher matcher(cv::NORM_L2, true);
	//std::vector<cv::DMatch> matches;
	//matcher.match(descriptors1,descriptors2,matches);
	//// draw the matches
	//cv::Mat imageMatches;
	////cv::drawMatches(image1, keypoints1, image2, keypoints2, matches,imageMatches, cv::Scalar(255,255,255), cv::Scalar(255, 255, 255), std::vector<char>(), 2);
	////cv::namedWindow("Matches");
	////cv::imshow("Matches", imageMatches);
	//// Convert keypoints into Point2f
	//std::vector<cv::Point2f> points1, points2;
	//for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
	//{
	//	float x = keypoints1[it->queryIdx].pt.x;
	//	float y = keypoints1[it->queryIdx].pt.y;
	//	points1.push_back(cv::Point2f(x,y));
	//	x = keypoints2[it->trainIdx].pt.x;
	//	y = keypoints2[it->trainIdx].pt.y;
	//	points2.push_back(cv::Point2f(x,y));
	//}
	//std::cout << "Number of matches: " << points2.size() << std::endl;
	//// Find the essential between image 1 and image 2
	//cv::Mat inliers;
	//cv::Mat essential = cv::findEssentialMat(points1, points2, cMatrix, cv::RANSAC, 0.9, 1.0, inliers);
	//int numberOfPts(cv::sum(inliers)[0]);
	//std::cout << "Number of inliers: " << numberOfPts << std::endl;
	//// draw the inlier matches
	//cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, imageMatches, cv::Scalar(255,255,255), cv::Scalar(255, 255, 255), inliers, 2);
	//cv::namedWindow("Inliers matches");
	//cv::imshow("Inliers matches", imageMatches);
	//// recover relative camera pose from essential matrix
	//cv::Mat rotation, translation;
	//cv::recoverPose(essential, points1, points2, cameraMatrix, rotation, translation, inliers);
	//std::cout << "rotation:" << rotation << std::endl;
	//std::cout << "translation:" << translation << std::endl;
	//// compose projection matrix from R,T 
	//cv::Mat projection2(3, 4, CV_64F);
	//rotation.copyTo(projection2(cv::Rect(0,0,3,3)));
	//translation.copyTo(projection2.colRange(3,4));
	//// compose generic projection matrix 
	//cv::Mat projection1(3,4,CV_64F,0.);
	//cv::Mat diag(cv::Mat::eye(3,3,CV_64F));
	//diag.copyTo(projection1(cv::Rect(0, 0, 3, 3)));
	//std::cout << "First Projection matrix=" << projection1 << std::endl;
	//std::cout << "Second Projection matrix=" << projection2 << std::endl;
	//// to contain the inliers
	//std::vector<cv::Vec2d> inlierPts1;
	//std::vector<cv::Vec2d> inlierPts2;
	//int j(0);
	//for (int i = 0; i < inliers.rows; i++)
	//{
	//	if (inliers.at<uchar>(i))
	//	{
	//		inlierPts1.push_back(cv::Vec2d(points1[i].x, points1[i].y));
	//		inlierPts2.push_back(cv::Vec2d(points2[i].x, points2[i].y));
	//	}
	//}
	//// undistort and normalize the image points
	//std::vector<cv::Vec2d> points1u;
	//cv::undistortPoints(inlierPts1, points1u, cameraMatrix, cameraDistCoeffs);
	//std::vector<cv::Vec2d> points2u;
	//cv::undistortPoints(inlierPts2, points2u, cameraMatrix, cameraDistCoeffs);
	//// triangulation
	//std::vector<cv::Vec3d> points3D;
	//triangulate(projection1, projection2, points1u, points2u, points3D);
	//// Create a viz window
	//cv::viz::Viz3d visualizer("Viz window");
	//visualizer.setBackgroundColor(cv::viz::Color::white());
	//// Construct the scene
	//// Create one virtual camera
	//cv::viz::WCameraPosition cam1(cMatrix,image1,1.0,cv::viz::Color::black());
	//// Create a second virtual camera
	//cv::viz::WCameraPosition cam2(cMatrix,image2,1.0,cv::viz::Color::black());
	//// choose one point for visualization
	//cv::Vec3d testPoint = triangulate(projection1, projection2, points1u[124], points2u[124]);
	//cv::viz::WSphere point3D(testPoint, 0.05, 10, cv::viz::Color::red());
	//double lenght(4.);
	//cv::viz::WLine line1(cv::Point3d(0,0,0),cv::Point3d(lenght*points1u[124](0), lenght*points1u[124](1), lenght), cv::viz::Color::green());
	//cv::viz::WLine line2(cv::Point3d(0., 0., 0.), cv::Point3d(lenght*points2u[124](0), lenght*points2u[124](1), lenght), cv::viz::Color::green());
	//// the reconstructed cloud of 3D points
	//cv::viz::WCloud cloud(points3D, cv::viz::Color::blue());
	//cloud.setRenderingProperty(cv::viz::POINT_SIZE, 3.);
	//// Add the virtual objects to the environment
	//visualizer.showWidget("Camera1", cam1);
	//visualizer.showWidget("Camera2", cam2);
	//visualizer.showWidget("Cloud", cloud);
	//visualizer.showWidget("Line1", line1);
	//visualizer.showWidget("Line2", line2);
	//visualizer.showWidget("Triangulated", point3D);
	//// Move the second camera	
	//cv::Affine3d pose(rotation, translation);
	//visualizer.setWidgetPose("Camera2", pose);
	//visualizer.setWidgetPose("Line2", pose);
	//while (!visualizer.wasStopped())
	//{
	//	visualizer.spinOnce(1,true);
	//}

	//// Read input images
	//cv::Mat image1 = cv::imread("brebeuf1.jpg", 0);
	//cv::Mat image2 = cv::imread("brebeuf2.jpg", 0);
	//if (!image1.data || !image2.data)
	//	return 0;
	//RobustMatcher rmatcher(cv::xfeatures2d::SIFT::create(250));
	//std::vector<cv::DMatch> matches;
	//std::vector<cv::KeyPoint> keypoints1, keypoints2;
	//cv::Mat fundamental = rmatcher.match(image1,image2,matches,keypoints1,keypoints2);
	//cv::Mat imageMatches;
	//cv::drawMatches(image1,keypoints1,image2,keypoints2,matches,imageMatches,cv::Scalar(255,255,255), cv::Scalar(255, 255, 255),std::vector<char>(),2);
	//cv::namedWindow("Matches");
	//cv::imshow("Matches", imageMatches);
	//// Convert keypoints into Point2f
	//std::vector<cv::Point2f> points1, points2;
	//for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
	//{
	//	float x = keypoints1[it->queryIdx].pt.x;
	//	float y = keypoints1[it->queryIdx].pt.y;
	//	points1.push_back(cv::Point2f(x,y));
	//	x = keypoints2[it->trainIdx].pt.x;
	//	y = keypoints2[it->trainIdx].pt.y;
	//	points2.push_back(cv::Point2f(x,y));
	//}
	//// Compute homographic rectification
	//cv::Mat h1, h2;
	//cv::stereoRectifyUncalibrated(points1,points2,fundamental,image1.size(),h1,h2);
	//// Rectify the images through warping
	//cv::Mat rectified1;
	//cv::warpPerspective(image1,rectified1,h1,image1.size());
	//cv::Mat rectified2;
	//cv::warpPerspective(image2, rectified2, h2, image1.size());
	//cv::namedWindow("Left Rectified Image");
	//cv::imshow("Left Rectified Image", rectified1);
	//cv::namedWindow("Right Rectified Image");
	//cv::imshow("Right Rectified Image", rectified2);
	//points1.clear();
	//points2.clear();
	//for (int i = 20; i < image1.rows - 20; i += 20)
	//{
	//	points1.push_back(cv::Point(image1.cols/2,i));
	//	points2.push_back(cv::Point(image2.cols / 2, i));
	//}
	//// Draw the epipolar lines
	//std::vector<cv::Vec3f> lines1;
	//cv::computeCorrespondEpilines(points1, 1, fundamental, lines1);
	//for (std::vector<cv::Vec3f>::const_iterator it = lines1.begin(); it != lines1.end(); ++it)
	//{
	//	cv::line(image2, cv::Point(0, -(*it)[2] / (*it)[1]), cv::Point(image2.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]), cv::Scalar(255, 255, 255));
	//}
	//std::vector<cv::Vec3f> lines2;
	//cv::computeCorrespondEpilines(points2, 2, fundamental, lines2);
	//for (std::vector<cv::Vec3f>::const_iterator it = lines2.begin();it != lines2.end(); ++it) 
	//{
	//	cv::line(image1, cv::Point(0, -(*it)[2] / (*it)[1]),cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image1.cols) / (*it)[1]),cv::Scalar(255, 255, 255));
	//}
	//cv::namedWindow("Left Epilines");
	//cv::imshow("Left Epilines", image1);
	//cv::namedWindow("Right Epilines");
	//cv::imshow("Right Epilines", image2);
	//// draw the pair
	//cv::drawMatches(image1, keypoints1,  // 1st image 
	//	image2, keypoints2,              // 2nd image 
	//	std::vector<cv::DMatch>(),
	//	imageMatches,		             // the image produced
	//	cv::Scalar(255, 255, 255),
	//	cv::Scalar(255, 255, 255),
	//	std::vector<char>(),
	//	2);
	//cv::namedWindow("A Stereo pair");
	//cv::imshow("A Stereo pair", imageMatches);
	//cv::Mat disparity;
	//cv::Ptr<cv::StereoMatcher> pStereo = cv::StereoSGBM::create(0,32,5);
	//pStereo->compute(rectified1,rectified2,disparity);
	//double minv, maxv;
	//disparity = disparity * 64;
	//cv::minMaxLoc(disparity, &minv, &maxv);
	//std::cout << minv << "+" << maxv << std::endl;
	//// Display the disparity map
	//cv::namedWindow("Disparity Map");
	//cv::imshow("Disparity Map", disparity);

	// draw the rectified pair
	/*
	cv::warpPerspective(image1, rectified1, h1, image1.size());
	cv::warpPerspective(image2, rectified2, h2, image1.size());
	cv::drawMatches(rectified1, keypoints1,  // 1st image
		rectified2, keypoints2,              // 2nd image
		std::vector<cv::DMatch>(),
		imageMatches,		                // the image produced
		cv::Scalar(255, 255, 255),
		cv::Scalar(255, 255, 255),
		std::vector<char>(),
		2);
	cv::namedWindow("Rectified Stereo pair");
	cv::imshow("Rectified Stereo pair", imageMatches);
	*/

	//// Open the video file
	//cv::VideoCapture capture("bike.avi");
	////cv::VideoCapture capture("http://www.laganiere.name/bike.avi");
	//// check if video successfully opened
	//if (!capture.isOpened())
	//	return 1;
	//// Get the frame rate
	//double rate = capture.get(cv::CAP_PROP_FPS);
	//std::cout << "Frame rate: " << rate << "fps" << std::endl;
	//long total = static_cast<long>(capture.get(CV_CAP_PROP_FRAME_COUNT));
	//std::cout << "Frame total: " << total << std::endl;
	//bool stop(false);
	//cv::Mat frame;
	//cv::namedWindow("Extracted Frame");
	//int delay = 1000 / rate;
	//long long i = 0;
	//std::string b = "bike";
	//std::string ext = ".bmp";
	//while (!stop)
	//{
	//	if (!capture.read(frame))
	//	{
	//		break;
	//	}
	//	cv::imshow("Extracted Frame", frame);
	//	std::string name(b);
	//	std::ostringstream ss;
	//	ss << std::setfill('0') << std::setw(3) << i;
	//	name += ss.str();
	//	i++;
	//	name += ext;
	//	std::cout << name << std::endl;
	//	cv::waitKey(delay);
	//	//if (cv::waitKey(delay) >= 0)
	//		//stop = true;
	//}
	//capture.release();
	//std::cout << "end" << std::endl;

	//VideoProcessor processor;
	//processor.setInput("bike.avi");
	//processor.displayInput("Input Video");
	//processor.dispalyOutput("Output Video");
	//processor.setDelay(1000. / processor.getFrameRate());
	//processor.setFrameProcessor(canny);
	//processor.setOutput("bikeCanny.avi", -1, 15);
	//processor.stopAtFrameNo(100);
	//processor.run();

	////cv::VideoCapture capture("bike.avi");
	//cv::VideoCapture capture(0);
	//if (!capture.isOpened())
	//	return 0;
	//cv::Mat frame;
	//cv::Mat foreground;
	//cv::Mat background;
	//cv::namedWindow("Extracted Foreground");
	//// The Mixture of Gaussian object
	//cv::Ptr<cv::BackgroundSubtractor> ptrMOG = cv::bgsegm::createBackgroundSubtractorMOG();
	//bool stop(false);
	//while (!stop)
	//{
	//	if (!capture.read(frame))
	//	{
	//		break;
	//	}
	//	// update the background and return the foreground
	//	ptrMOG->apply(frame,foreground,0.01);
	//	cv::threshold(foreground, foreground, 128, 255, cv::THRESH_BINARY_INV);
	//	cv::imshow("Extracted Foreground", foreground);
	//	cv::waitKey(10);
	//}

	/*VideoProcessor processor;
	BGFGSegmentor segmentor;
	segmentor.setThreshold(25);
	processor.setInput("bike.avi");
	processor.displayInput("Input Frame");
	processor.setFrameProcessor(&segmentor);
	processor.displayOutput("Extracted Foreground");
	processor.setDelay(1000. / processor.getFrameRate());
	processor.run();*/

	//VideoProcessor processor;
	//FeatureTracker tracker;
	//processor.setInput("bike.avi");
	//processor.setFrameProcessor(&tracker);
	//processor.displayOutput("Tracked Features");
	//processor.setDelay(1000. / processor.getFrameRate());
	//processor.stopAtFrameNo(100);
	//processor.run();

	//cv::Mat frame1 = cv::imread("goose/goose230.bmp", 0);
	//cv::Mat frame2 = cv::imread("goose/goose237.bmp", 0);
	//cv::Mat combined(frame1.rows, frame1.cols + frame2.cols, CV_8U);
	///*frame1.copyTo(combined.colRange(0, frame1.cols));
	//frame2.copyTo(combined.colRange(frame1.cols, frame1.cols + frame2.cols));
	//cv::imshow("Frames", combined);*/
	//// Create the optical flow algorithm
	//cv::Ptr<cv::DualTVL1OpticalFlow> tvl1 = cv::createOptFlow_DualTVL1();
	//std::cout << "regularization coeeficient: " << tvl1->getLambda() << std::endl; // the smaller the soomther
	//std::cout << "Number of scales: " << tvl1->getScalesNumber() << std::endl; // number of scales
	//std::cout << "Scale step: " << tvl1->getScaleStep() << std::endl; // size between scales
	//std::cout << "Number of warpings: " << tvl1->getWarpingsNumber() << std::endl; // size between scales
	//std::cout << "Stopping criteria: " << tvl1->getEpsilon() << " and " << tvl1->getOuterIterations() << std::endl; // size between scales
	//cv::Mat oflow;
	//tvl1->calc(frame1, frame2, oflow);
	//cv::Mat flowImage;
	//drawOpticalFlow(oflow,flowImage, 8, 2, cv::Scalar(0,0,0));
	//cv::imshow("Optical Flow", flowImage);
	//// Draw the optical flow image
	//cv::Mat flowImage2;
	//drawOpticalFlow(oflow,     // input flow vectors 
	//	flowImage2, // image to be generated
	//	8,         // display vectors every 8 pixels
	//	2,         // multiply size of vectors by 2
	//	cv::Scalar(0, 0, 0)); // vector color
	//cv::imshow("Smoother Optical Flow", flowImage2);

	//VideoProcessor processor;
	//std::vector<std::string> imgs;
	//std::string prefix = "goose/goose";
	//std::string ext = ".bmp";
	//for (long i = 130; i < 317; i++)
	//{
	//	std::string name(prefix);
	//	std::ostringstream ss;
	//	ss << std::setfill('0') << std::setw(3) << i;
	//	name += ss.str();
	//	name += ext;
	//	std::cout << name << std::endl;
	//	imgs.push_back(name);
	//}
	//// Create feature tracker instance
	////VisualTracker tracker(cv::TrackerMedianFlow::createTracker());
	//VisualTracker tracker(cv::TrackerKCF::createTracker());
	//processor.setInput(imgs);
	//processor.setFrameProcessor(&tracker);
	//processor.displayOutput("Tracked object");
	//processor.setDelay(50);
	//cv::Rect bb(290, 100, 65, 40);
	//tracker.setBoundingBox(bb);
	//processor.run();

	//cv::Mat image = imread("girl.jpg", cv::IMREAD_GRAYSCALE);
	////cv::imshow("Original image", image);
	////cv::Mat lbpImage;
	////lbp(image, lbpImage);
	////cv::imshow("LBP image", lbpImage);
	//cv::Ptr<cv::face::FaceRecognizer> recognizer = cv::face::createLBPHFaceRecognizer(1, 8, 8, 8, 200.);
	//std::vector<cv::Mat> referenceImages;
	//std::vector<int> labels;
	//referenceImages.push_back(cv::imread("face0_1.png", cv::IMREAD_GRAYSCALE));
	//labels.push_back(0); // person 0
	//referenceImages.push_back(cv::imread("face0_2.png", cv::IMREAD_GRAYSCALE));
	//labels.push_back(0); // person 0
	//referenceImages.push_back(cv::imread("face1_1.png", cv::IMREAD_GRAYSCALE));
	//labels.push_back(1); // person 1
	//referenceImages.push_back(cv::imread("face1_2.png", cv::IMREAD_GRAYSCALE));
	//labels.push_back(1); // person 1
	////// the 4 positive samples
	////cv::Mat faceImages(2 * referenceImages[0].rows, 2 * referenceImages[0].cols, CV_8U);
	////for (int i = 0; i < 2; i++)
	////{
	////	for (int j = 0; j < 2; j++)
	////	{
	////		referenceImages[i * 2 + j].copyTo(faceImages(cv::Rect(j*referenceImages[i * 2 + j].cols, i*referenceImages[i * 2 + j].rows, referenceImages[i * 2 + j].cols, referenceImages[i * 2 + j].rows)));
	////	}
	////}
	////cv::resize(faceImages, faceImages, cv::Size(), 0.5, 0.5);
	////cv::imshow("Reference faces", faceImages);
	////// train the recognizer
	//recognizer->train(referenceImages, labels);
	//int predictedLabel = -1;
	//double confidence = 0.0;
	////// Extract a face image
	//cv::Mat inputImage;
	//cv::resize(image(cv::Rect(160, 75, 90, 90)), inputImage, cv::Size(256, 256));
	//cv::imshow("Input image", inputImage);
	////// predict the label of this image
	//recognizer->predict(inputImage, predictedLabel, confidence);
	//std::cout << "Image label= " << predictedLabel << " (" << confidence << ")" << std::endl;



	cv::Mat image = imread("girl.jpg", cv::IMREAD_GRAYSCALE);
cv::Ptr<cv::face::FaceRecognizer> recognizer = cv::face::createLBPHFaceRecognizer(1, 8, 8, 8, 200.);
std::vector<cv::Mat> referenceImages;
std::vector<int> labels;
referenceImages.push_back(cv::imread("face0_1.png", cv::IMREAD_GRAYSCALE));
labels.push_back(0); // person 0
referenceImages.push_back(cv::imread("face0_2.png", cv::IMREAD_GRAYSCALE));
labels.push_back(0); // person 0
referenceImages.push_back(cv::imread("face1_1.png", cv::IMREAD_GRAYSCALE));
labels.push_back(1); // person 1
referenceImages.push_back(cv::imread("face1_2.png", cv::IMREAD_GRAYSCALE));
labels.push_back(1); // person 1
//// the 4 positive samples
//cv::Mat faceImages(2 * referenceImages[0].rows, 2 * referenceImages[0].cols, CV_8U);
//for (int i = 0; i < 2; i++)
//{
//	for (int j = 0; j < 2; j++)
//	{
//		referenceImages[i * 2 + j].copyTo(faceImages(cv::Rect(j*referenceImages[i * 2 + j].cols, i*referenceImages[i * 2 + j].rows, referenceImages[i * 2 + j].cols, referenceImages[i * 2 + j].rows)));
//	}
//}
//cv::resize(faceImages, faceImages, cv::Size(), 0.5, 0.5);
//cv::imshow("Reference faces", faceImages);
//// train the recognizer
recognizer->train(referenceImages, labels);
int predictedLabel = -1;
double confidence = 0.0;
//// Extract a face image
cv::Mat inputImage;
cv::resize(image(cv::Rect(160, 75, 90, 90)), inputImage, cv::Size(256, 256));
cv::imshow("Input image", inputImage);
//// predict the label of this image
recognizer->predict(inputImage, predictedLabel, confidence);
std::cout << "Image label= " << predictedLabel << " (" << confidence << ")" << std::endl;




	// open the positive sample images
	//std::vector<cv::Mat> referenceImages;
	//referenceImages.push_back(cv::imread("stopSamples/stop00.png"));
	//referenceImages.push_back(cv::imread("stopSamples/stop01.png"));
	//referenceImages.push_back(cv::imread("stopSamples/stop02.png"));
	//referenceImages.push_back(cv::imread("stopSamples/stop03.png"));
	//referenceImages.push_back(cv::imread("stopSamples/stop04.png"));
	//referenceImages.push_back(cv::imread("stopSamples/stop05.png"));
	//referenceImages.push_back(cv::imread("stopSamples/stop06.png"));
	//referenceImages.push_back(cv::imread("stopSamples/stop07.png"));
	//// create a composite image
	//cv::Mat positiveImages(2 * referenceImages[0].rows, 4 * referenceImages[0].cols, CV_8UC3);
	//for (int i = 0; i < 2; i++)
	//{
	//	for (int j = 0; j < 4; j++)
	//	{
	//		referenceImages[i * 2 + j].copyTo(positiveImages(cv::Rect(j*referenceImages[i * 2 + j].cols, i*referenceImages[i * 2 + j].rows, referenceImages[i * 2 + j].cols, referenceImages[i * 2 + j].rows)));
	//	}
	//}
	//cv::imshow("Positive samples", positiveImages);
	//cv::Mat inputImage = cv::imread("stopSamples/stop9.jpg");
	//cv::resize(inputImage, inputImage, cv::Size(), 0.5, 0.5);
	//cv::CascadeClassifier cascade;
	//if (!cascade.load("stopSamples/classifier/cascade.xml"))
	//{
	//	std::cout << "Error when loading the cascade classfier!" << std::endl;
	//	return -1;
	//}
	//// predict the label of this image
	//std::vector<cv::Rect> detections;
	//cascade.detectMultiScale(inputImage, detections, 1.1, 1, 0, cv::Size(48, 48), cv::Size(128, 128));
	//std::cout << "detections= " << detections.size() << std::endl;
	//for (int i = 0; i < detections.size(); i++)
	//	cv::rectangle(inputImage, detections[i], cv::Scalar(255, 255, 255), 2);
	//cv::imshow("Stop sign detection", inputImage);

	////// Detecting faces
	//cv::Mat picture = cv::imread("girl.jpg");
	//std::vector<cv::Rect> detections;
	//cv::CascadeClassifier faceCascade;
	//if (!faceCascade.load("haarcascade_frontalface_default.xml"))
	//{
	//	std::cout << "Error when loading the face cascade classfier!" << std::endl;
	//	return -1;
	//}
	//faceCascade.detectMultiScale(picture, detections, 1.1, 3, 0, cv::Size(48, 48), cv::Size(128, 128));
	//std::cout << "detections = " << detections.size() << std::endl;
	//// draw detections on image
	//for (int i = 0; i < detections.size(); i++)
	//	cv::rectangle(picture, detections[i], cv::Scalar(255, 255, 255), 2);
	//// Detecting eyes
	//cv::CascadeClassifier eyeCascade;
	//std::vector<cv::Rect> detectionseye;
	//if (!eyeCascade.load("haarcascade_eye.xml"))
	//{
	//	std::cout << "Error when loading the eye cascade classfier!" << std::endl;
	//	return -1;
	//}
	//eyeCascade.detectMultiScale(picture, detectionseye, 1.1, 3, 0, cv::Size(24, 24), cv::Size(64, 64));
	//std::cout << "detectionseye = " << detectionseye.size() << std::endl;
	//// draw detections on image
	//for (int i = 0; i < detectionseye.size(); i++)
	//	cv::rectangle(picture, detectionseye[i], cv::Scalar(0, 0, 0), 2);
	//cv::imshow("Detection results", picture);

	
	//cv::VideoCapture capture(0);
	//if (!capture.isOpened())
	//{
	//	std::cout << "can not open the carmea" << std::endl;
	//}

	////cv::Mat picture = cv::imread("girl.jpg");
	//cv::Mat picture;
	//std::vector<cv::Rect> detections;
	//cv::CascadeClassifier faceCascade;
	//if (!faceCascade.load("haarcascade_frontalface_default.xml"))
	//{
	//	std::cout << "Error when loading the face cascade classfier!" << std::endl;
	//	return -1;
	//}
	//// Detecting eyes
	//cv::CascadeClassifier eyeCascade;
	//std::vector<cv::Rect> detectionseye;
	//if (!eyeCascade.load("haarcascade_eye.xml"))
	//{
	//	std::cout << "Error when loading the eye cascade classfier!" << std::endl;
	//	return -1;
	//}

	//while (1) {
	//	if (!capture.read(picture))
	//	{
	//		break;
	//	}
	//	//faceCascade.detectMultiScale(picture, detections, 1.1, 3, 0, cv::Size(48, 48), cv::Size(128, 128));
	//	faceCascade.detectMultiScale(picture, detections, 1.1, 3, 0, cv::Size(68, 68), cv::Size(150, 150));
	//	std::cout << "detections = " << detections.size() << std::endl;
	//	// draw detections on image

	//	if (detections.size() > 1)
	//	{
	//		//too many people
	//		cv::putText(picture, "Excessive number of people in the horizon", cv::Point(10,100), cv::FONT_HERSHEY_PLAIN, 2.0, 255, 2);
	//	}
	//	else if(detections.size() == 1)
	//	{
	//		cv::Mat faceROI(picture, detections[0]);

	//		cv::rectangle(picture, detections[0], cv::Scalar(255, 255, 255), 2);

	//		eyeCascade.detectMultiScale(faceROI, detectionseye, 1.1, 3, 0, cv::Size(24, 24), cv::Size(64, 64));
	//		std::cout << "detectionseye = " << detectionseye.size() << std::endl;
	//		// draw detections on image
	//		for (int i = 0; i < detectionseye.size(); i++)
	//			cv::rectangle(picture, cv::Rect(detections[0].x + detectionseye[i].x, detections[0].y + detectionseye[i].y, detectionseye[i].width, detectionseye[i].height), cv::Scalar(0, 0, 0), 2);
	//	}


		//for (int i = 0; i < detections.size(); i++)
		//{
		//	cv::Mat faceROI(picture, detections[i]);

		//	cv::rectangle(picture, detections[i], cv::Scalar(255, 255, 255), 2);

		//	eyeCascade.detectMultiScale(faceROI, detectionseye, 1.1, 3, 0, cv::Size(24, 24), cv::Size(64, 64));
		//	std::cout << "detectionseye = " << detectionseye.size() << std::endl;
		//	// draw detections on image
		//	for (int j = 0; j < detectionseye.size(); j++)
		//		cv::rectangle(picture, cv::Rect(detections[i].x + detectionseye[j].x, detections[i].y + detectionseye[j].y, detectionseye[j].width, detectionseye[j].height), cv::Scalar(0, 0, 0), 2);
		//}

		
		//cv::imshow("Detection results", picture);




		
	/*	cv::waitKey(500);
	}*/

	//cv::Mat image = imread("girl.jpg", cv::IMREAD_GRAYSCALE);
	//cv::imshow("Original image", image);
	//cv::HOGDescriptor hog(cv::Size((image.cols/16)*16, (image.rows/16)*16),cv::Size(16,16), cv::Size(16,16), cv::Size(4,4),9);
	//std::vector<float> descriptors;
	//// Draw a representation of HOG cells
	//cv::Mat hogImage = image.clone();
	//drawHOGDescriptors(image, hogImage, cv::Size(16, 16), 9);
	//cv::imshow("HOG image", hogImage);
	//// generate the filename
	//std::vector<std::string> imgs;
	//std::string prefix = "stopSamples/stop";
	//std::string ext = ".png";
	//// loading 8 positive samples
	//std::vector<cv::Mat> positives;
	//for (long i = 0; i < 8; i++)
	//{
	//	std::string name(prefix);
	//	std::ostringstream ss;
	//	ss << std::setfill('0') << std::setw(2) << i;
	//	name += ss.str();
	//	name += ext;
	//	positives.push_back(cv::imread(name, cv::IMREAD_GRAYSCALE));
	//}
	//// the first 8 positive samples
	//cv::Mat posSamples(2 * positives[0].rows, 4 * positives[0].cols, CV_8U);
	//for (int i = 0; i < 2; i++)
	//	for (int j = 0; j < 4; j++) {
	//		positives[i * 4 + j].copyTo(posSamples(cv::Rect(j*positives[i * 4 + j].cols, i*positives[i * 4 + j].rows, positives[i * 4 + j].cols, positives[i * 4 + j].rows)));
	//	}
	//cv::imshow("Positive samples", posSamples);
	//// loading 8 negative samples
	//std::string nprefix = "stopSamples/neg";
	//std::vector<cv::Mat> negatives;
	//for (long i = 0; i < 8; i++)
	//{
	//	std::string name(nprefix);
	//	std::ostringstream ss;
	//	ss << std::setfill('0') << std::setw(2) << i;
	//	name += ss.str();
	//	name += ext;
	//	negatives.push_back(cv::imread(name, cv::IMREAD_GRAYSCALE));
	//}
	//// the first 8 negative samples
	//cv::Mat negSamples(2 * negatives[0].rows, 4 * negatives[0].cols, CV_8U);
	//for (int i = 0; i < 2; i++)
	//	for (int j = 0; j < 4; j++) {
	//		negatives[i * 4 + j].copyTo(negSamples(cv::Rect(j*negatives[i * 4 + j].cols, i*negatives[i * 4 + j].rows, negatives[i * 4 + j].cols, negatives[i * 4 + j].rows)));
	//	}
	//cv::imshow("Negative samples", negSamples);
	//// The HOG descriptor for stop sign detection
	//cv::HOGDescriptor hogDesc(positives[0].size(),cv::Size(8,8),cv::Size(4,4),cv::Size(4,4),9);
	//// compute first descriptor 
	//std::vector<float> desc;
	//hogDesc.compute(positives[0], desc);
	//std::cout << "Positive sample size: " << positives[0].rows << "x" << positives[0].cols << std::endl;
	//std::cout << "HOG descriptor size: " << desc.size() << std::endl;
	//// the matrix of sample descriptors
	//int featureSize = desc.size();
	//int numberOfSamples = positives.size() + negatives.size();
	//// create the matrix that will contain the samples HOG
	//cv::Mat samples(numberOfSamples, featureSize, CV_32FC1);
	//// fill first row with first descriptor
	//for (int i = 0; i < featureSize; i++)
	//{
	//	samples.ptr<float>(0)[i] = desc[i];
	//}
	//// compute descriptor of the positive samples
	//for (int j = 1; j < positives.size(); j++)
	//{
	//	hogDesc.compute(positives[j],desc);
	//	// fill the next row with current descriptor
	//	for (int i = 0; i < featureSize; i++)
	//	{
	//		samples.ptr<float>(j)[i] = desc[i];
	//	}
	//}
	//// compute descriptor of the negative samples
	//for (int j = 0; j < negatives.size(); j++)
	//{
	//	hogDesc.compute(negatives[j], desc);
	//	// fill the next row with current descriptor
	//	for (int i = 0; i < featureSize; i++)
	//	{
	//		samples.ptr<float>(j + positives.size())[i] = desc[i];
	//	}
	//}
	//// Create the labels
	//cv::Mat labels(numberOfSamples, 1, CV_32SC1);
	//// labels of positive samples
	//labels.rowRange(0, positives.size()) = 1.0;
	//// labels of negative samples
	//labels.rowRange(positives.size(), numberOfSamples) = -1.0;
	//// create SVM classifier
	//cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	//svm->setType(cv::ml::SVM::C_SVC);
	//svm->setKernel(cv::ml::SVM::LINEAR);
	//// prepare the training data
	//cv::Ptr<cv::ml::TrainData> trainingData = cv::ml::TrainData::create(samples, cv::ml::SampleTypes::ROW_SAMPLE, labels);
	//// SVM training
	//svm->train(trainingData);
	//cv::Mat queries(4, featureSize, CV_32FC1);
	//// fill the rows with query descriptors
	//hogDesc.compute(cv::imread("stopSamples/stop08.png", cv::IMREAD_GRAYSCALE), desc);
	//for (int i = 0; i < featureSize; i++)
	//	queries.ptr<float>(0)[i] = desc[i];
	//hogDesc.compute(cv::imread("stopSamples/stop09.png", cv::IMREAD_GRAYSCALE), desc);
	//for (int i = 0; i < featureSize; i++)
	//	queries.ptr<float>(1)[i] = desc[i];
	//hogDesc.compute(cv::imread("stopSamples/neg08.png", cv::IMREAD_GRAYSCALE), desc);
	//for (int i = 0; i < featureSize; i++)
	//	queries.ptr<float>(2)[i] = desc[i];
	//hogDesc.compute(cv::imread("stopSamples/neg09.png", cv::IMREAD_GRAYSCALE), desc);
	//for (int i = 0; i < featureSize; i++)
	//	queries.ptr<float>(3)[i] = desc[i];
	//cv::Mat predictions;
	//// Test the classifier with the last two pos and neg samples
	//svm->predict(queries, predictions);
	//for (int i = 0; i < 4; i++)
	//{
	//	std::cout << "query: " << i << ": " << ((predictions.at<float>(i) < 0.0) ? "Negative" : "Positive") << std::endl;
	//}

	//// People detection
	//cv::Mat myImage = cv::imread("person.jpg", cv::IMREAD_GRAYSCALE);
	//// create the detector
	//std::vector<cv::Rect> peoples;
	//cv::HOGDescriptor peopleHog;
	//peopleHog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
	//// detect peoples
	//peopleHog.detectMultiScale(myImage, peoples, 0, cv::Size(4,4), cv::Size(32,32),1.1,2);
	//// draw detections on image
	//std::cout << "Number of peoples detected: " << peoples.size() << std::endl;
	//for (int i = 0; i < peoples.size(); i++)
	//{
	//	cv::rectangle(myImage, peoples[i], cv::Scalar(255,255,255),2);
	//}
	//cv::imshow("People detection", myImage);

//std::unordered_set<std::string> number;
//std::default_random_engine randnum;
//std::uniform_int_distribution<unsigned> u(0, 9);
//while (number.size() <= 9)
//{
//	std::string tem = std::to_string(u(randnum));
//	if (number.find(tem) == number.end())
//	{
//		number.insert(tem);
//		std::cout << tem << " ";
//	}
//}
//	std::string tem;
//	std::unordered_set<std::string>::iterator it = number.begin();
//	while (it != number.end())
//	{
//		tem += *it;
//		it++;
//	}
//	std::cout << tem;


	cv::waitKey();
	return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门提示: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
