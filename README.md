# blob-detection
Code for live blob detection from a webcam camera feed in OpenCV.

# Using the code.
This code is written in C++. After cloning/pulling, compile it using:

```
g++ blob.cpp -o BlobTracker `pkg-config --cflags --libs opencv`

```

This should generate an executable named BlobTracker, which you can then run.
Note that this assumes that you have the opencv library already installed.

# Selecting which camera to use

At the beginning of the main function, there is a VideoCapture command that tells OpenCV to start reading data from a camera:

```
VideoCapture cap(1);

```
The number in parentheses tells OpenCV which camera you want to use. If it is set to 0, it will use the default camera (usually your computer's built-in webcam). If set to 1, it will use the next available camera plugged into a USB port.

# Changing the thresholds

As written, the code allows the user to filter possible blob candidates by Threshold (color difference with surroundings), Area (# of pixels filled up), Circularity (self-explanatory), Convexity, and Inertia. These flags can be set to true or false. If true, the values to filter by can be set as well.

# Centroid detection

The code now allows for centroid detection. In each frame, it will average out the x and y locations of the keypoints in order to determine the centroid, and will then plot said centroid as a blue filled circle on the image in real time. Test it out using the following images:

![Dark Circle](https://github.com/enginerd887/blob-detection/blob/master/Reverse%20Circle.jpg) ![Reverse Circle](https://github.com/enginerd887/blob-detection/blob/master/hvgqv.png)

Note that you may have to adjust the minimum captured pixel area using the sliders.
