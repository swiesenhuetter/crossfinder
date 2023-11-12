import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt


def cross_center(outline, indices):
    """
    Finds the center of the cross by averaging the 4 max concave points
    :param outline:
    :return:
    """
    x = outline[indices][:,0].mean()
    y = outline[indices][:,1].mean()
    return x, y


def draw_convexity_max_points(img, point_indixes, contour): 
    """
    Draws the convexity max points on the image
    :param img:
    :param point_indixes: in the contour
    :param contour:
    :return:
    """
    for i in point_indixes:
        cv2.circle(img, (contour[i][0][0], contour[i][0][1]), 2, 0, -1)
    return img


image = cv2.imread('data/typical.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('original image', image)

thresh = 20
canny_output = cv2.Canny(image, thresh, thresh * 2)

cv2.imshow('canny image', canny_output)

contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

# only keep if more than 3 points
contours = [cont for cont in contours if len(cont) > 20]

#create an empty image for contours
img_contours = np.zeros(canny_output.shape)


# draw the contours on the empty image
cv2.drawContours(img_contours, contours, -1, 255, 1)
cv2.imshow('contour image', img_contours)

img_simpler = np.ones(canny_output.shape)

for i, cont in enumerate(contours):
    rect = cv2.minAreaRect(cont)
    # if rectange area is too small, skip

    area = rect[1][0] * rect[1][1]
    if 200 > area or area > 10000:
        continue

    # if rectangle is too narrow, skip
    if rect[1][0] < 6 or rect[1][1] < 6:
        continue

    # if rotation angle is not a multiple of 45 +/- 2, skip
    if 1 < abs(rect[2]) % 45 < 44:
        print(f"skipping {rect[2]} : {abs(rect[2] % 45)}")
        continue

    box = cv2.boxPoints(rect)
    cv2.fillConvexPoly(img_simpler, np.int32(box), 0)


cv2.imshow('simpler mage', img_simpler)

cv2.imwrite('data/detected.jpg', (255 * img_simpler))

cv2.waitKey(0)
cv2.destroyAllWindows()
