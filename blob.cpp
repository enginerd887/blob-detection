#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iterator>


using namespace cv;  // Computer vision namespace
using namespace std; // Standard C++ namespace

////////////////////////// Program Setup //////////////////////////////////////

// Function Declarations
void on_low_r_thresh_trackbar(int, void *);
void on_high_r_thresh_trackbar(int, void *);
void on_low_g_thresh_trackbar(int, void *);
void on_high_g_thresh_trackbar(int, void *);
void on_low_b_thresh_trackbar(int, void *);
void on_high_b_thresh_trackbar(int, void *);
string type2str(int type);


void CallBackFunc(int event, int x, int y, int flags, void* userdata);

// Globals (May need to scan through this and delete stuff)
int low_r=30, low_g=30, low_b=30;
int high_r=100, high_g=100, high_b=100;

const int minTmax = 50;
const int maxTmax = 1000;
const int areaMax = 2000;
const int convexMax = 100;
const int circleMax = 100;
const int inertiaMax = 100;

int minTslider = 20;
int maxTslider = 1000;
int areaSlider = 100;
int convexSlider = 1;
int circleSlider = 3;
int inertiaSlider = 1;

int Imultiply = 60;
int threshVal = 100;
double multiplier = 60;
double threshValue = 100;

int oldSize = 0;
int mouseClicked = 0;


/////////////////////// Blob Detector Setup ///////////////////////////////////

// Setup SimpleBlobDetector parameters.
SimpleBlobDetector::Params params;
cv::Ptr<cv::SimpleBlobDetector> detector;

void display_vector(const vector<int> &v) //note the const
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

  multiplier = (double)Imultiply;
  threshValue = (double)threshVal;
}

/////////////////////////// Main Program ///////////////////////////////////////
int main( int argc, char** argv )
{

  /////////////////////// Open Camera, Setup Stuff /////////////////////////////

  VideoCapture cap(1); // open the camera (opens default cam 0 otherwise)
  //cvSetCaptureProperty(cap,CV_CAP_PROP_AUTO_EXPOSURE,0.0);
  cap.set(CV_CAP_PROP_AUTO_EXPOSURE,0.0);
  cap.set(CV_CAP_PROP_EXPOSURE,50);



  if(!cap.isOpened())  // check if we succeeded
      return -1;


  // Storage for blobs
  std::vector<KeyPoint> keypointList;
  std::vector<KeyPoint> keypointList2;


  // Create windows
  namedWindow("keypoints",1);
  namedWindow("threshold",1);
  //namedWindow("threshold2",1);

  //Create TrackBars for keypoints parameters
  char minTname[30];
  char maxTname[30];
  char areaName[30];
  char circleName[30];
  char convexName[30];
  char inertiaName[30];

  char multiplierName[30];
  char thresholdName[30];

  sprintf(minTname,"Min. Threshold");
  sprintf(maxTname,"Max. Threshold");
  sprintf(areaName,"Min. Pixel Area");
  sprintf(circleName,"Min. Circularity");
  sprintf(convexName, "Min. Convexity");
  sprintf(inertiaName,"Min. Inertia Ratio");
  sprintf(multiplierName,"Image Multiplier");
  sprintf(thresholdName,"Gray Threshold");

  createTrackbar(minTname,"keypoints",&minTslider,minTmax,on_trackbar );
  createTrackbar(maxTname,"keypoints",&maxTslider,maxTmax,on_trackbar );
  createTrackbar(areaName,"keypoints",&areaSlider,areaMax,on_trackbar );
  createTrackbar(circleName,"keypoints",&circleSlider,circleMax,on_trackbar );
  createTrackbar(convexName,"keypoints",&convexSlider,convexMax,on_trackbar );
  createTrackbar(inertiaName,"keypoints",&inertiaSlider,inertiaMax,on_trackbar );

  createTrackbar(multiplierName,"threshold",&Imultiply,200,on_trackbar );
  createTrackbar(thresholdName,"threshold",&threshVal,255,on_trackbar );

      //-- Trackbars to set thresholds for RGB values
  //createTrackbar("Low R","threshold", &low_r, 255, on_low_r_thresh_trackbar);
  //createTrackbar("High R","threshold", &high_r, 255, on_high_r_thresh_trackbar);
  //createTrackbar("Low G","threshold", &low_g, 255, on_low_g_thresh_trackbar);
  //createTrackbar("High G","threshold", &high_g, 255, on_high_g_thresh_trackbar);
  //createTrackbar("Low B","threshold", &low_b, 255, on_low_b_thresh_trackbar);
  //createTrackbar("High B","threshold", &high_b, 255, on_high_b_thresh_trackbar);

  // Will be used later to hold starting centroid of blobs
  Point2f b(0.0,0.0);
  Point startPoint = b;
  int counter = 0;

  Mat im;
  Mat imAverage;
  Mat imCurrent;
  Mat filtered;
  Mat filtered2;
  Mat ref;

  // Capture the reference images

  cap >> im;
  cap >> imAverage;
  ref = Mat::zeros(im.size(),CV_32FC3);

  int counter2 = 1;

  // Set the mouse CallBackFunc
  setMouseCallback("threshold",CallBackFunc,NULL);
  int AvgCounter = 0;
  double xAccum = 0.0;
  double yAccum = 0.0;
  double alpha = .5;
  ////////////////////////// The Infinite Loop ////////////////////////////////
  while (1)
  {

      double min,max;

      // The following commands are not used here but may be useful

            //cvtColor(im,im,COLOR_RGB2GRAY); Converts to grayscale images
            //absdiff(im,ref,filtered); Gives the differenced image between two images
            //bitwise_not(filtered,filtered); Inverts image pixels
            //inRange(im,Scalar(low_b,low_g,low_r), Scalar(high_b,high_g,high_r),filtered); Color thresholding
            //adaptiveThreshold(im,im,150,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,11,12); Adaptive thresholding

      ////////////////// Read new image, compare to reference //////////////////

      if (mouseClicked == 1)
      {
        cap >> ref;
        mouseClicked = 0;
      }

      cap >> im; // get a new frame from camera

      //accumulateWeighted(im,im,0.5);
      if (AvgCounter < 20){
        AvgCounter++;
        add(imAverage,im,imAverage);
        divide(imAverage,2,imAverage);
      }

      subtract(im,imAverage,imCurrent);

      // Divide out the reference background image, normalize result
      filtered = imCurrent*multiplier/imAverage;
      minMaxLoc(filtered,&min,&max);
      filtered = filtered - min;
      filtered = (filtered/(max-min))*255;

      cvtColor(imCurrent,imCurrent,COLOR_RGB2GRAY);
      threshold(imCurrent,filtered,threshValue,255,0);
      bitwise_not(filtered,filtered);
      imshow("keypoints",imCurrent);
      imshow("threshold",filtered);


      on_trackbar(areaSlider, 0);

      //////////////// Detect and Display Blobs ////////////////////////////////
      detector->detect( filtered, keypointList );
      //detector->detect( filtered2, keypointList2);
      // Draw detected blobs as red circles.
      // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
      // the size of the circle corresponds to the size of blob

      Mat im_with_keypoints;
      //Mat im_with_refpoints;
      drawKeypoints( im, keypointList, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
      //drawKeypoints( im, keypointList2,im_with_refpoints, Scalar(255,0,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
      //addWeighted(im_with_keypoints,.5,im_with_refpoints,.5,0.0,im_with_keypoints);
      // Show blobs
      imshow("keypoints", im_with_keypoints );
      imshow("threshold",filtered);
      //imshow("threshold2",filtered2);
      //imshow("Original",im);


      //////////////////// Calculate Blob Centroid /////////////////////////////
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
        xAccum = (alpha*xAvg)+(1-alpha)*xAccum;
        yAccum = (alpha*yAvg)+(1-alpha)*yAccum;

        // Output the centroid
        //cout << xAvg << " " << yAvg << endl;

        // Convert centroid values to an OpenCV "Point"
        Point2f a(xAccum,yAccum);
        Point centroidVal = a;


        //On the first few iterations through, capture the centroid
        if (counter < 100){
          startPoint = centroidVal;
          counter++;

        }
        // Afterwards, lock the starting centroid

        // Draw a filled blue circle at the centroidVal

        //syntax:
        //circle(image,center point, size, color, fill, lineType)
        circle(im_with_keypoints,centroidVal,10.0,Scalar(255,0,0),-1,8);
        //circle(im_with_keypoints,startPoint,10.0,Scalar(100,150,50),-1,8);

        //Draw the circle to the screen
        imshow("keypoints", im_with_keypoints);
        oldSize = keypointList.size();
      }



      if(waitKey(40) >= 0) break;
  }

  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}

//////////////////////////// Assist Functions /////////////////////////////////

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
     if  ( event == EVENT_LBUTTONDOWN )
     {
       mouseClicked = 1;
       cout << "Reference Image Reset" << endl;
     }
     else if  ( event == EVENT_RBUTTONDOWN )
     {
          cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
     }
     else if  ( event == EVENT_MBUTTONDOWN )
     {
          cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
     }

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

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}
