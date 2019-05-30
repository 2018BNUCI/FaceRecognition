from skimage import feature
from skimage.filters import gabor
import numpy as np
import matplotlib.pyplot as plt
import cv2

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self,image,eps=1e-7):
        # plt.subplot(1, 3, 1)
        # img = cv2.cvtColor(image,cv2.CV_32S)
        # plt.imshow(img)
        # plt.subplot(1, 3, 2)
        # fr,fi = gabor(image,frequency=0.6)
        # plt.imshow(fi)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(image,self.numPoints,self.radius,method='uniform')
        # plt.subplot(1, 2 ,1),plt.xticks([]),plt.yticks([]),plt.title('lbp image')
        # plt.imshow(lbp,plt.cm.gray)
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins = np.arange(0,self.numPoints+3),
                                 range = (0,self.numPoints+2))
        # print(hist)
        hist = hist.astype("float")
        hist/=(hist.sum()+eps)
        # plt.subplot(1,2,2),plt.xticks([]),plt.yticks([]),plt.title('histogram')
        # plt.hist(hist)
        # plt.show()
        return hist



