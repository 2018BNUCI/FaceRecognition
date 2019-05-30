import numpy as np
import cv2
import matplotlib.pyplot as plt

class gaborFilter:
    def build_filters(self):
        filters = []
        ksize = range(7,18,2)
        lamda = np.pi/2.0
        for theta in np.arange(0, np.pi, np.pi/4):
            for K in range(6):
                kern = cv2.getGaborKernel((ksize[K],ksize[K]),1.0,theta,lamda,0.5, 0,ktype = cv2.CV_32F)
                kern /= 1.5*kern.sum()
                filters.append(kern)
        return filters

    def process(self, img ,filters):
        acc = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img,cv2.CV_8UC3,kern)
            np.maximum(acc,fimg,acc)
        return acc

    def getGabor(self, img, filters):
        res = []
        for i in range(len(filters)):
            res1 = self.process(img, filters[i])
            res.append(np.asarray(res1))

        # 展示res
        # for temp in range(len(res)):
        #     plt.subplot(4,6,temp+1),plt.xticks([]),plt.yticks([])
        #     plt.imshow(res[temp], plt.cm.gray)
        #
        # plt.show()
        return res

# if __name__ == '__main__':
#     gf = gaborFilter()
#     filters = gf.build_filters()
#     imagePath = 'C:/Users/User/Documents/test/2.jpg'
#     image = cv2.imread(imagePath)
#     gf.getGabor(image, filters)

# def deginrad(degree):
#     radiant = 2*np.pi/360*degree
#     return radiant
#
# img = cv2.imread('C:/Users/User/PycharmProjects/bnuface/getLBP/img/lena.bmp')

# image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#
# theta = deginrad(25)
# g_kernel = cv2.getGaborKernel((4,4),10,theta,5,0,0,ktype=cv2.CV_32F)
#
# fgimg = cv2.filter2D(image,cv2.CV_8UC3,g_kernel)
# cv2.imshow('gabor image',fgimg)
# #cv2.waitKey(0)
# plt.imshow(fgimg)
# plt.show()