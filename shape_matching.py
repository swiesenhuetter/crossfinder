import cv2
import numpy as np

stroke = "m 530.25383,637.73084 -8.58913,-9.24264 30.12298,-28.08932 -30.19313,-29.39546 9.16865,-8.65929 29.1242,28.42436 29.13889,-28.11759 9.78282,10.08513 -28.92925,28.93182 28.42356,27.86243 -10.28853,8.50638 -28.62585,-28.67761 -29.64457,28.37179"
# create a polyline from svg stoke string

movements = stroke.split(' ')
movements = [p.split(',') for p in movements[1:]]
movements = [(float(p[0]), float(p[1])) for p in movements]

abs_points = [(100, 100)]

for i in range(1, len(movements)):
    abs_points.append((abs_points[-1][0] + movements[i][0], abs_points[-1][1] + movements[i][1]))

# abs_points to numpy array
abs_points = np.array(abs_points, dtype=np.int32)


# create an empty image
img = np.ones((1000, 1000, 3), np.uint8) * 255

# draw the abs polyline on the image

# for i in range(1, len(abs_points)):
#     cv2.line(img, (int(abs_points[i-1][0]), int(abs_points[i-1][1])), (int(abs_points[i][0]), int(abs_points[i][1])), (0, 0, 0), 1)
#
# fill polyline
cv2.fillPoly(img, np.array([abs_points], dtype=np.int32), (0, 0, 0))


def calculate_shape_similarity(contour1, contour2):
    return cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I2, 0.0)

camera_image = cv2.imread('data/typical.png', cv2.IMREAD_GRAYSCALE)

thresh = 20
canny_output = cv2.Canny(camera_image, thresh, thresh * 2)



shape_mask = np.zeros(camera_image.shape, np.uint8)

# find contours in the image
contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

# draw the contours on the original image
cv2.drawContours(shape_mask, contours, -1, 255, 1)


best_match_contour = min(contours, key=lambda contour: calculate_shape_similarity(contour, abs_points))

x, y, w, h = cv2.boundingRect(best_match_contour)
cv2.rectangle(shape_mask, (x, y), (x + w, y + h), 255, 2)

cv2.imshow('camera image', shape_mask)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
