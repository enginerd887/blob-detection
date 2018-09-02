#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include <algorithm>
#include <iterator>


using namespace cv;
using namespace std;


void on_low_r_thresh_trackbar(int, void *);
void on_high_r_thresh_trackbar(int, void *);
void on_low_g_thresh_trackbar(int, void *);
void on_high_g_thresh_trackbar(int, void *);
void on_low_b_thresh_trackbar(int, void *);
void on_high_b_thresh_trackbar(int, void *);
int low_r=30, low_g=30, low_b=30;
int high_r=100, high_g=100, high_b=100;

// Globals
const int minTmax = 50;
const int maxTmax = 1000;
const int areaMax = 1000;
const int convexMax = 100;
const int circleMax = 100;
const int inertiaMax = 100;

int minTslider = 20;
int maxTslider = 1000;
int areaSlider = 100;
int convexSlider = 87;
int circleSlider = 3;
int inertiaSlider = 1;

int oldSize = 0;
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
  namedWindow("threshold",1);

  //Create TrackBars for keypoints
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

  namedWindow("Object Detection", WINDOW_NORMAL);
      //-- Trackbars to set thresholds for RGB values
  createTrackbar("Low R","threshold", &low_r, 255, on_low_r_thresh_trackbar);
  createTrackbar("High R","threshold", &high_r, 255, on_high_r_thresh_trackbar);
  createTrackbar("Low G","threshold", &low_g, 255, on_low_g_thresh_trackbar);
  createTrackbar("High G","threshold", &high_g, 255, on_high_g_thresh_trackbar);
  createTrackbar("Low B","threshold", &low_b, 255, on_low_b_thresh_trackbar);
  createTrackbar("High B","threshold", &high_b, 255, on_high_b_thresh_trackbar);

  // Will be used later to hold starting centroid of blobs
  Point2f b(0.0,0.0);
  Point startPoint = b;
  int counter = 0;

  while (1)
  {
      Mat im;
      Mat filtered;

      cap >> im; // get a new frame from camera
      //cvtColor(im,im,COLOR_RGB2GRAY);
      //adaptiveThreshold(im,im,150,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,11,12);
      //threshold(im,im,100,200,0);
      inRange(im,Scalar(low_b,low_g,low_r), Scalar(high_b,high_g,high_r),filtered);
      bitwise_not(filtered,filtered);
      GaussianBlur(im, im, Size(7,7), 1.5, 1.5);
      GaussianBlur(filtered,filtered,Size(7,7),1.5,1.5);
      on_trackbar(areaSlider, 0);
      // Detect blobs
      detector->detect( filtered, keypointList );

      // Draw detected blobs as red circles.
      // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
      // the size of the circle corresponds to the size of blob

      Mat im_with_keypoints;
      drawKeypoints( im, keypointList, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

      // Show blobs
      imshow("keypoints", im_with_keypoints );
      imshow("threshold",filtered);
      //imshow("Original",im);

      // Check if there are any keypoints in view
      if (keypointList.size() > 0){

        double xSum = 0;
        double ySum = 0;

        // Add up all the x and y values respectively
        for( size_t ii = 0; ii < keypointList.size( ); ++ii ){
          xSum += keypointList[ii].pt.x;
          ySum += keypointList[ii].pt.y;
          //cout << keypointList[ii].size << endl;
        }
        // Divide by number of keypoints to get centroid of blobs
        double xAvg = xSum/keypointList.size();
        double yAvg = ySum/keypointList.size();

        // Output the centroid
        //cout << xAvg << " " << yAvg << endl;

        // Convert centroid values to an OpenCV "Point"
        Point2f a(xAvg,yAvg);
        Point centroidVal = a;


        //On the first few iterations through, capture the centroid
        if (counter < 100){
          startPoint = centroidVal;
          counter++;
        //  cout << "caught Start!" << endl;
        }
        // Afterwards, lock the starting centroid

        // Draw a filled blue circle at the centroidVal

        //syntax:
        //circle(image,center point, size, color, fill, lineType)
        circle(im_with_keypoints,centroidVal,10.0,Scalar(100,255,100),-1,8);
        circle(im_with_keypoints,startPoint,10.0,Scalar(100,150,50),-1,8);

        //Draw the circle to the screen
        imshow("keypoints", im_with_keypoints);
        oldSize = keypointList.size();
      }



      if(waitKey(40) >= 0) break;
  }

  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}


void on_low_r_thresh_trackbar(int, void *)
{
    low_r = min(high_r-1, low_r);
    setTrackbarPos("Low R","threshold", low_r);
}
void on_high_r_thresh_trackbar(int, void *)
{
    high_r = max(high_r, low_r+1);
    setTrackbarPos("High R", "threshold", high_r);
}
void on_low_g_thresh_trackbar(int, void *)
{
    low_g = min(high_g-1, low_g);
    setTrackbarPos("Low G","threshold", low_g);
}
void on_high_g_thresh_trackbar(int, void *)
{
    high_g = max(high_g, low_g+1);
    setTrackbarPos("High G", "threshold", high_g);
}
void on_low_b_thresh_trackbar(int, void *)
{
    low_b= min(high_b-1, low_b);
    setTrackbarPos("Low B","threshold", low_b);
}
void on_high_b_thresh_trackbar(int, void *)
{
    high_b = max(high_b, low_b+1);
    setTrackbarPos("High B", "threshold", high_b);
}
