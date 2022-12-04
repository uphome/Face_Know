import cv2
import PyQt5
import numpy as np

image = cv2.imread('/home/hu/图片/Screenshot_2022-12-04-18-14-30-31_e39d2c7de19156b0683cd93e8735f348.jpg')
# 图片大小为(1200, 540）   使用3种颜色表示  BGR
cv2.namedWindow('image_ray', cv2.WINDOW_NORMAL)  # 进行窗口缩放   以及窗口的调整
image_ray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(image,100,200)
# cv2.imshow('edges',edges)
# cv2.imshow('image', image)
image_ray=cv2.bilateralFilter(image_ray,4,75,75)

i, j = 0, 0
for i in range(image_ray.shape[0]):
    for j in range(image_ray.shape[1]):
        if int(image_ray[i, j]) >= 190  and int(image_ray[i,j])<=200:
            image_ray[i,j]=0
        if int(image_ray[i,j])>=250:
            image_ray[i,j]=255

cv2.imshow('image_ray', image_ray)
cv2.waitKey(0)
cv2.destroyAllWindows()
