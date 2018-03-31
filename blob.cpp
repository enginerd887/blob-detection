#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{

  VideoCapture cap(1); // open the default camera
  if(!cap.isOpened())  // check if we succeeded
      return -1;

  // Declare a named window for the camera image
  Mat edges;
  namedWindow("edges",1);

  // Setup SimpleBlobDetector parameters.
  SimpleBlobDetector::Params params;

  // Change thresholds
  params.minThreshold = 20;
  params.maxThreshold = 1000;

  // Filter by Area.
  params.filterByArea = true;
  params.minArea = 400;

  // Filter by Circularity
  params.filterByCircularity = false;
  params.minCircularity = 0.3;

  // Filter by Convexity
  params.filterByConvexity = true;
  params.minConvexity = 0.87;

  // Filter by Inertia
  params.filterByInertia = true;
  params.minInertiaRatio = 0.01;

  cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
  // Storage for blobs
  std::vector<KeyPoint> keypoints;

  for(;;)
  {
      Mat im;
      cap >> im; // get a new frame from camera
      cvtColor(im, edges, COLOR_BGR2GRAY);
      GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);

      // Detect blobs
      detector->detect( im, keypoints );

      // Draw detected blobs as red circles.
      // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
      // the size of the circle corresponds to the size of blob

      Mat im_with_keypoints;
      drawKeypoints( im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

      // Show blobs
      imshow("keypoints", im_with_keypoints );

      if(waitKey(30) >= 0) break;
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
