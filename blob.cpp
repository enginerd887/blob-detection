#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include <algorithm>
#include <iterator>


using namespace cv;
using namespace std;

// Globals
const int minTmax = 50;
const int maxTmax = 1000;
const int areaMax = 1000;
const int convexMax = 100;
const int circleMax = 100;
const int inertiaMax = 100;

int minTslider = 20;
int maxTslider = 1000;
int areaSlider = 400;
int convexSlider = 87;
int circleSlider = 3;
int inertiaSlider = 1;
// Setup SimpleBlobDetector parameters.
SimpleBlobDetector::Params params;
cv::Ptr<cv::SimpleBlobDetector> detector;

//note the const
void display_vector(const vector<int> &v)
{
    std::copy(v.begin(), v.end(),
        std::ostream_iterator<int>(std::cout, " "));
}

// Function for when trackbar changes

void on_trackbar(int,void*)
{
  // Change thresholds
  params.minThreshold = (double)minTslider;
  params.maxThreshold = (double)maxTslider;

  // Filter by Area.
  params.filterByArea = true;
  params.minArea = (double)areaSlider;

  // Filter by Circularity
  params.filterByCircularity = true;
  params.minCircularity = (double)circleSlider/circleMax;

  // Filter by Convexity
  params.filterByConvexity = true;
  params.minConvexity = (double)convexSlider/convexMax;

  // Filter by Inertia
  params.filterByInertia = false;
  params.minInertiaRatio = (double)inertiaSlider/inertiaMax;

  detector = cv::SimpleBlobDetector::create(params);
}

int main( int argc, char** argv )
{

  VideoCapture cap(1); // open the default camera
  if(!cap.isOpened())  // check if we succeeded
      return -1;


  // Storage for blobs
  std::vector<KeyPoint> keypointList;

  // Create window
  namedWindow("keypoints",1);

  //Create TrackBar
  char minTname[30];
  char maxTname[30];
  char areaName[30];
  char circleName[30];
  char convexName[30];
  char inertiaName[30];

  sprintf(minTname,"Min. Threshold");
  sprintf(maxTname,"Max. Threshold");
  sprintf(areaName,"Min. Pixel Area");
  sprintf(circleName,"Min. Circularity");
  sprintf(convexName, "Min. Convexity");
  sprintf(inertiaName,"Min. Inertia Ratio");

  createTrackbar(minTname,"keypoints",&minTslider,minTmax,on_trackbar );
  createTrackbar(maxTname,"keypoints",&maxTslider,maxTmax,on_trackbar );
  createTrackbar(areaName,"keypoints",&areaSlider,areaMax,on_trackbar );
  createTrackbar(circleName,"keypoints",&circleSlider,circleMax,on_trackbar );
  createTrackbar(convexName,"keypoints",&convexSlider,convexMax,on_trackbar );
  createTrackbar(inertiaName,"keypoints",&inertiaSlider,inertiaMax,on_trackbar );


  while (1)
  {
      Mat im;
      cap >> im; // get a new frame from camera
      GaussianBlur(im, im, Size(7,7), 1.5, 1.5);
      on_trackbar(areaSlider, 0);
      // Detect blobs
      detector->detect( im, keypointList );

      // Draw detected blobs as red circles.
      // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
      // the size of the circle corresponds to the size of blob

      Mat im_with_keypoints;
      drawKeypoints( im, keypointList, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

      // Show blobs
      imshow("keypoints", im_with_keypoints );

      // Check if there are any keypoints in view
      if (keypointList.size() > 0){
        double xSum = 0;
        double ySum = 0;

        // Add up all the x and y values respectively
        for( size_t ii = 0; ii < keypointList.size( ); ++ii ){
          xSum += keypointList[ii].pt.x;
          ySum += keypointList[ii].pt.y;
        }
        // Divide by number of keypoints to get centroid of blobs
        double xAvg = xSum/keypointList.size();
        double yAvg = ySum/keypointList.size();

        // Output the centroid
        cout << xAvg << " " << yAvg << endl;

        // Convert centroid values to an OpenCV "Point"
        Point2f a(xAvg,yAvg);
        Point centroidVal = a;

        // Draw a filled blue circle at the centroidVal

        //syntax:
        //circle(image,center point, size, color, fill, lineType)
        circle(im_with_keypoints,centroidVal,10.0,Scalar(100,255,100),-1,8);

        //Draw the circle to the screen
        imshow("keypoints", im_with_keypoints);
      }



      if(waitKey(40) >= 0) break;
  }

  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
