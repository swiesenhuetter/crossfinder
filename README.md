# crossfinder
An OpenCV example: How to detect low contrast  cross shapes in an image. We need their center 2D coordinate (in image pixel coordinates).

![result image](https://github.com/swiesenhuetter/crossfinder/blob/master/low-contrast-x.png?raw=true)

The idea is to firsrt extract the contours of all features. Then we extract straight lines in those feature bundaries by applying the Ramer Douglas Peucker Algorithm. This leaves us simplified polyline shapes approximating the countours with line segments.
Finally, we select shapes with certain known geometrical properties (number and position of vertices). In those shapes we calculate the center and draw and save results in an image file.

![result image](https://github.com/swiesenhuetter/crossfinder/blob/master/detected.jpg?raw=true)