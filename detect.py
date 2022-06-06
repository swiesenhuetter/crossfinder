import cv2
import numpy as np


def cross_center(outline):
    """
    Finds the center of the cross
    :param outline:
    :return:
    """
    mid = int(outline.shape[0] / 2)
    quart = int(mid / 2)

    # Find the center of the outline
    x1 = (outline[0][0] + outline[mid][0]) / 2.0
    y1 = (outline[0][1] + outline[mid][1]) / 2.0
    x2 = (outline[quart][0] + outline[mid+quart][0]) / 2.0
    y2 = (outline[quart][1] + outline[mid+quart][1]) / 2.0
    return (x1+x2)/2, (y1+y2)/2


image = cv2.imread('data/low-contrast-x.png', cv2.IMREAD_GRAYSCALE)

thresh = 10
ret, thresh_img = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#create an empty image for contours
img_contours = np.zeros(thresh_img.shape)
# draw the contours on the empty image
cv2.drawContours(img_contours, contours, -1, 255, 20)
#save image

cv2.imshow('contour mage', img_contours)
cv2.imshow('contour mage', image)

img_simpler = np.ones(thresh_img.shape)

for cont in contours:
    peri = cv2.arcLength(cont, True)
    simpler = cv2.approxPolyDP(cont, 0.03 * peri, True)
    silen = cv2.arcLength(simpler, True)

    x_hull = simpler.reshape((len(simpler), 2))
    
    if len(x_hull) == 8 or len(x_hull) == 12:
        x, y = cross_center(x_hull)
        cv2.polylines(img_simpler, [x_hull], True, 0, 1)
        cv2.putText(img_simpler, "x: " + str(x) + "y: " + str(y), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        cv2.circle(img_simpler, (int(x), int(y)), 5, 0, -1)
    else:
        cv2.polylines(img_simpler, [x_hull], True, 0, 1)

cv2.imshow('simpler mage', img_simpler)

cv2.waitKey(0)
cv2.destroyAllWindows()