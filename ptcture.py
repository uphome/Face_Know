import cv2
import numpy as np



img = cv2.imread('/home/hu/图片/Screenshot_2022-12-04-16-18-59-69_e39d2c7de19156b0683cd93e8735f348.jpg')
img_ray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

bulr=cv2.bilateralFilter(img,4,75,75)

hsv = cv2.cvtColor(bulr, cv2.COLOR_BGR2HSV)

low_b=np.array([255,0,0])
high_b=np.array([245,255,245])
mask=cv2.inRange(hsv,low_b,high_b)


res = cv2.bitwise_and(img,img, mask= mask)




cv2.namedWindow('image_ray', cv2.WINDOW_NORMAL)
cv2.imshow('image_ray', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
