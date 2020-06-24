# Canny Edge

# Hough Line Transform
1. lines: A vector that will store the parameters (r,\theta) of the detected lines
2. rho: The resolution of the parameter r in pixels. We use 1 pixel.
3. theta: The resolution of the parameter \theta in radians. We use 1 degree (CV_PI/180)
4. threshold: The minimum number of intersections to “detect” a line
5. srn and stn: Default parameters to zero. Check OpenCV reference for more info.

# Probabilistic Hough Line Transform
1. lines: A vector that will store the parameters (x_{start}, y_{start}, x_{end}, y_{end}) of the detected lines
2. rho: The resolution of the parameter r in pixels. We use 1 pixel.
3. theta: The resolution of the parameter \theta in radians. We use 1 degree (CV_PI/180)
4. threshold: The minimum number of intersections to “detect” a line
5. minLinLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
6. maxLineGap: The maximum gap between two points to be considered in the same line.
* https://www.learnopencv.com/hough-transform-with-opencv-c-python/

# Resize

# Thresholding

# Pespective Transform
