import cv2
import numpy as np
import matplotlib.pyplot as plt
imgPath2 = 'C:/Users/User/PycharmProjects/bnuface/getLBP/img/barbara.bmp'
imgPath1 = 'C:/Users/User/Documents/test/1.jpg'
imgPath3 = 'C:/Users/User/Documents/test/10.jpg'
imgOri = cv2.imread(imgPath1)
imgYuv = cv2.cvtColor(imgOri,cv2.COLOR_BGR2YUV)
imgYuv[:,:,0] = cv2.equalizeHist(imgYuv[:,:,0])
img = cv2.cvtColor(imgYuv, cv2.COLOR_YUV2BGR)
cv2.imwrite(imgPath3,img)
# plt.subplot(2,2,2)
# plt.imshow(img)
# plt.title('equalized image')
# plt.xticks([])
# plt.yticks([])

imgYcc = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
row,col,channels = img.shape

imgSki = np.zeros(img.shape, np.uint8)
imgSki = img.copy()


def check1(R,G,B,Y,Cr,Cb):
    if R > G and R > B:
        if (G >= B and 5 * R - 12 * G + 7 * B >= 0) or (G < B and 5 * R + 7 * G - 12 * B >= 0):
            if Cr > 135 and Cr < 180 and Cb > 85 and Cb < 135 and Y > 80:
                return True
    return False

def check2(Cr,Cb):
    if Cr>137 and Cr<175 and Cb>100 and Cb<118:
        return True
    return False

for r in range(row):
    for c in range(col):
        f = 0
        R = img.item(r,c,0)
        G = img.item(r,c,1)
        B = img.item(r,c,2)

        Y = imgYcc.item(r,c,0)
        Cr = imgYcc.item(r,c,1)
        Cb = imgYcc.item(r,c,2)

        if check2(Cr,Cb):
            f = 1

        if f == 0:
            imgSki.itemset((r,c,0),0)
            imgSki.itemset((r,c,2),0)
            imgSki.itemset((r,c,1),0)
        else:
            imgSki.itemset((r, c, 0), 255)
            imgSki.itemset((r, c, 2), 255)
            imgSki.itemset((r, c, 1), 255)

plt.subplot(1,2,1)
plt.imshow(img)
plt.title('origin image')
plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2)
plt.imshow(imgSki)
plt.title('YCrCb Skin Image')
plt.xticks([]),plt.yticks([])

plt.show()
