# crossfinder
An OpenCV example: How to detect low contrast  cross shapes in an image. We need their center 2D coordinate (in image pixel coordinates).

![input image](data/low-contrast-x.png?raw=true)

The idea is to first extract the contours of all features. 

![contours image](data/contours.jpg?raw=true)

Then we extract straight lines in those feature boundaries by applying the Ramer Douglas Peucker Algorithm. This leaves us with simplified polyline shapes approximating the countours with line segments.
Finally, we select shapes with certain known geometrical properties (number and position of vertices). In those shapes we calculate the center and draw and save results in an image file.

![result image](data/detected.jpg?raw=true)

