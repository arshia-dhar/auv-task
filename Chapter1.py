import cv2
import numpy as np

img=cv2.imread("Resources\Arrow1.jpeg")

HSV=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lowerRed = np.array([0, 50, 20])
upperRed = np.array([5, 255, 255])
mask = cv2.inRange(HSV, lowerRed, upperRed)

kernel = np.ones((5,5))
imgDilate = cv2.dilate(mask, kernel, iterations=2)
imgErode = cv2.erode(imgDilate, kernel, iterations=1)

contours, heirarchy= cv2.findContours(imgErode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    corners = len(approx)

    if 4 <= corners <= 8:
        cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)

points = cv2.goodFeaturesToTrack(imgErode,7,0.01,10)
points = np.int64(points)

xmax, ymax = (np.max(points, axis=0)).ravel()
xmin, ymin = (np.min(points, axis=0)).ravel()

if (abs(xmax - xmin) < abs(ymax - ymin)):
    if (np.count_nonzero(points[:, 0, 0] == xmax) == 2):
        print('RIGHT')
    else:
        print('LEFT')
else:
    if (np.count_nonzero(points[:, 0, 1] == ymax) == 2):
        print('DOWN')
    else:
        print('UP')

cv2.imshow('Arrow Detection', img)
cv2.waitKey(0)
