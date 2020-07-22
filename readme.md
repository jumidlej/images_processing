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
1. Interpolation

# Thresholding

# Pespective Transform

# Soldas
72 102
110 102
168 92
194 92
239 92
149 113
195 126
241 128
87 149
113 149
87 175
113 175
176 201
285 200
322 220
239 319
273 319
298 320
334 332
301 356
66 365
92 365
140 370
210 371
248 371
275 371
86 393
320 404
86 412
221 414
247 414
275 414
320 428
294 453
320 450
324 481
158 468
194 468
156 493
194 493
231 489
255 486
232 519
256 519
321 520
294 544
180 566
206 594
67 594
176 613
203 615
279 609
318 609
315 655
98 671
142 687
174 684
202 684
226 676
319 680