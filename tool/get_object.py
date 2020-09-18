import cv2
import numpy as np

img = cv2.imread("xuehua1.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_img, 30, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
img_contours = sorted(img_contours, key=cv2.contourArea)
area = {}
for i,k in enumerate(img_contours):
    area[cv2.contourArea(k)] = k
ka = area[max(area.keys())]
mask = np.zeros(img.shape[:2], np.uint8)
cv2.drawContours(mask, [area[max(area.keys())]],-1, 255, -1)
cv2.imwrite("mask.png",mask)
new_img = cv2.bitwise_and(img, img, mask=mask)
cv2.imwrite("cvextract2.png",new_img)