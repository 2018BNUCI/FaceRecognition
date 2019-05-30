import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/User/PycharmProjects/bnuface/getLBP/img/lena.bmp')
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=1500)

keypoints_sift, descriptors = sift.detectAndCompute(img,None)

img = cv2.drawKeypoints(img,keypoints_sift,None)

cv2.imshow("SIFT",img)
cv2.waitKey(0)
cv2.destroyAllWindows()