#include "pch.h"
#include "CameraCalibrator.h"

// Open chessboard images and extract corner points
int CameraCalibrator::addChessboardPoints(const std::vector<std::string>& filelist, cv::Size& boardSize, std::string windowName)
{
	// the points on the chessboard
	std::vector<cv::Point2f> imageCorners;
	std::vector<cv::Point3f> objectCorners;

	// 3D Scene Points: Initialize the chessboard corners in the chessboard reference frame
	// The corners are at 3D location (X,Y,Z)= (i,j,0)
	for (int i = 0; i < boardSize.height; i++)
	{
		for (int j = 0; j < boardSize.width; j++)
		{
			objectCorners.push_back(cv::Point3f(i,j,0.0f));
		}
	}
	// 2D Image points to contain chessboard image
	cv::Mat image;
	int successes = 0;
	// for all viewpoints
	for (int i = 0; i < filelist.size(); i++)
	{
		image = cv::imread(filelist[i],0);
		// Get the chessboard corners
		bool found = cv::findChessboardCorners(image,boardSize,imageCorners);
		// Get subpixel accuracy on the corners
		if (found)
		{
			cv::cornerSubPix(image,imageCorners,cv::Size(5, 5),cv::Size(-1, -1),cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS,30,0.1));
			// If we have a good board, add it to our data
			if (imageCorners.size() == boardSize.area())
			{
				// Add image and scene points from one view
				addPoints(imageCorners, objectCorners);
				successes++;
			}
		}
		if (windowName.length() > 0 && imageCorners.size() == boardSize.area())
		{
			//Draw the corners
			cv::drawChessboardCorners(image, boardSize, imageCorners, found);
			cv::imshow(windowName, image);
			cv::waitKey(100);
		}
	}
	return successes;
}

// Add scene points and corresponding image points
void CameraCalibrator::addPoints(const std::vector<cv::Point2f>& imageCorners, const std::vector<cv::Point3f>& objectCorners)
{
	// 2D image points from one view
	imagePoints.push_back(imageCorners);
	// corresponding 3D scene points
	objectPoints.push_back(objectCorners);
}

// Calibrate the camera
// returns the re-projection error
double CameraCalibrator::calibrate(const cv::Size imageSize)
{
	// undistorter must be reinitialized
	mustInitUndistort = true;
	//Output rotations and translations
	std::vector<cv::Mat> rvecs, tvecs;
	// start calibration
	//return cv::calibrateCamera((objectPoints,imagePoints,imageSize,cameraMatrix,distCoeffs,rvecs,tvecs,flag, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, DBL_EPSILON));
	return cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs,flag);
}



// remove distortion in an image (after calibration)
cv::Mat CameraCalibrator::remap(const cv::Mat& image, cv::Size& outputSize)
{
	cv::Mat undistorted;
	if (outputSize.height == -1)
		outputSize = image.size();
	if (mustInitUndistort)
	{
		cv::initUndistortRectifyMap(cameraMatrix,distCoeffs,cv::Mat(),cv::Mat(),outputSize, CV_32FC1, map1, map2);
		mustInitUndistort = false;
	}
	// Apply mapping functions
	cv::remap(image,undistorted,map1,map2,cv::INTER_LINEAR);
	return undistorted;
}

// Set the calibration options

void CameraCalibrator::setCalibrationFlag(bool radial8CoeffEnabled, bool tangentialParamEnabled)
{
	// Set the flag used in cv::calibrateCamera()
	flag = 0;
	if(!tangentialParamEnabled)
		flag += CV_CALIB_ZERO_TANGENT_DIST;
	if (radial8CoeffEnabled) 
		flag += CV_CALIB_RATIONAL_MODEL;
}