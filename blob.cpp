#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

// Globals
const int areaMax = 1000;
int areaSlider;
double minArea;

// Setup SimpleBlobDetector parameters.
SimpleBlobDetector::Params params;
cv::Ptr<cv::SimpleBlobDetector> detector;

// Function for when trackbar changes

void on_trackbar(int,void*)
{
  // Change thresholds
  params.minThreshold = 20;
  params.maxThreshold = 1000;

  // Filter by Area.
  params.filterByArea = true;
  params.minArea = (double)areaSlider;

  // Filter by Circularity
  params.filterByCircularity = false;
  params.minCircularity = 0.3;

  // Filter by Convexity
  params.filterByConvexity = true;
  params.minConvexity = 0.87;

  // Filter by Inertia
  params.filterByInertia = true;
  params.minInertiaRatio = 0.01;

  detector = cv::SimpleBlobDetector::create(params);
}

int main( int argc, char** argv )
{

  VideoCapture cap(1); // open the default camera
  if(!cap.isOpened())  // check if we succeeded
      return -1;


  // Storage for blobs
  std::vector<KeyPoint> keypoints;

  // Create window
  namedWindow("keypoints",1);

  //Create TrackBar
  char areaName[20];
  sprintf(areaName,"Min. Pixel Area");
  createTrackbar(areaName,"keypoints",&areaSlider,areaMax,on_trackbar );

  for(;;)
  {
      Mat im;
      cap >> im; // get a new frame from camera
      GaussianBlur(im, im, Size(7,7), 1.5, 1.5);
      on_trackbar(areaSlider, 0);
      // Detect blobs
      detector->detect( im, keypoints );

      // Draw detected blobs as red circles.
      // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
      // the size of the circle corresponds to the size of blob

      Mat im_with_keypoints;
      drawKeypoints( im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

      // Show blobs
      imshow("keypoints", im_with_keypoints );
      cout << minArea << " " << params.minArea << endl;
      if(waitKey(30) >= 0) break;
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
