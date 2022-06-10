import cv2
import numpy as np


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


image = cv2.imread('data/low-contrast-x.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('original image', image)

thresh = 10
ret, thresh_img = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#create an empty image for contours
img_contours = np.zeros(thresh_img.shape)
# draw the contours on the empty image
cv2.drawContours(img_contours, contours, -1, 255, 1)
cv2.imwrite('data/contours.jpg', (255 * img_contours))
cv2.imshow('contour mage', img_contours)

img_simpler = np.ones(thresh_img.shape)

for cont in contours:
    peri = cv2.arcLength(cont, True)
    simpler = cv2.approxPolyDP(cont, 0.05 * peri, True)
    silen = cv2.arcLength(simpler, True)

    x_hull = simpler.reshape((len(simpler), 2))

    convex_hull_pts = cv2.convexHull(x_hull, returnPoints = True)
    cv2.polylines(img_simpler, [convex_hull_pts], True, 0, 1)
    if len(x_hull) > 4:
        convex_hull = cv2.convexHull(x_hull, returnPoints = False)
        try:
            convexityDefects = cv2.convexityDefects(simpler, convex_hull)
        except Exception as e:
            print(e)
            continue
        x = x_hull[0][0]
        y = x_hull[0][1]
        cv2.putText(img_simpler, "def {}".format((len(convexityDefects))), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

        max_conv_points = convexityDefects.T[2][0]
        draw_convexity_max_points(img_simpler, max_conv_points, simpler)


        if len(convexityDefects) == 4:
            x, y = cross_center(x_hull, max_conv_points)
            cv2.polylines(img_simpler, [x_hull], True, 0, 1)
            cv2.putText(img_simpler, "x: " + str(x) + "y: " + str(y), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
            cv2.circle(img_simpler, (int(x), int(y)), 3, 0, -1)
        else:
            cv2.polylines(img_simpler, [x_hull], True, 0, 1)

cv2.imshow('simpler mage', img_simpler)

cv2.imwrite('data/detected.jpg', (255 * img_simpler))

cv2.waitKey(0)
cv2.destroyAllWindows()