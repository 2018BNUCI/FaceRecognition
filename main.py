import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
import matplotlib.pyplot as plt
import cv2
from HaarCascadeClassifier import haarcascadeclassifier
from localBinaryPatterns import LocalBinaryPatterns
from gabor_filter import gaborFilter
from grabCuts import grabcut
def Opt(r):
    (a,b,c,d) = r
    k = 7
    if a-c/k >= 0:
        a = a-c/k
    else: a = 0
    if b - d/k >=0:
        b = b-d/k
    else: b = 0
    c = c + c*2/k
    d = d + d*2/k
    b-=c/2/k
    a = int(a)
    b = int(b)
    c = int(c)
    d = int(d)
    r = (a,b,c,d)
    return r

def drawRect(image, r):
    (x,y,w,h) = r
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    return image

imagePath = 'C:/Users/User/Documents/test/22.jpg'
imgOrigin = cv2.imread(imagePath)
imgRGB = cv2.cvtColor(imgOrigin, cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1)
plt.imshow(imgRGB),plt.title('Origin image'),plt.xticks([]),plt.yticks([])

# equalize histogram
imgYuv = cv2.cvtColor(imgOrigin,cv2.COLOR_BGR2YUV)
imgYuv[:,:,0] = cv2.equalizeHist(imgYuv[:,:,0])
imgOrigin = cv2.cvtColor(imgYuv,cv2.COLOR_YUV2BGR)


imgRGB = cv2.cvtColor(imgOrigin, cv2.COLOR_BGR2RGB)
imgGray = cv2.cvtColor(imgOrigin, cv2.COLOR_BGR2GRAY)

hcc = haarcascadeclassifier()
faces = hcc.haarface(imgGray)

for rect in faces:
    rect = Opt(rect)
    (x,y,w,h) = rect
    roi_gray = imgGray[y:y+h,x:x+w]
    imgCopy = np.zeros(imgRGB.shape,np.uint8)
    imgCopy = imgRGB.copy()
    roi = imgCopy[y:y+h,x:x+w]
    eyes = hcc.haareye(roi_gray)
    (x1, y1, w1, h1) = eyes[0]
    (x2, y2, w2, h2) = eyes[1]
    sin_angle = (y1 + h1 / 2.0 - y2 - h2 / 2.0) / (x1 + w1 - x2 - w2 / 2.0)
    eye_h =h -( 0.5 * ((y1 + 0.5 * h1) + (y2 + 0.5 * h2)))
    cv2.circle(imgRGB, (x1,y1), 10, (255, 0, 0), 8)
    Geo_char_p = eye_h / h  # 几何特征p
    print("Geo_char_p = " , Geo_char_p)

    noses = hcc.haarnose(roi_gray)
    (x3, y3, w3, h3) = noses[0]
    nose_h = h -( y3 + 0.5 * h3)
    Geo_char_q = (h - nose_h) / nose_h  # 几何特征q
    print("Geo_char_q = " , Geo_char_q)
    geocascade = [Geo_char_p,Geo_char_q]

    gb = grabcut(rect=rect,count=3)
    imgGrab = gb.gcresult(imgRGB)
    # plt.imshow(imgGrab)
    gf = gaborFilter()
    desc = LocalBinaryPatterns(24, 8)
    filters = gf.build_filters()
    roirgb = cv2.cvtColor(roi_gray,cv2.COLOR_GRAY2RGB)
    res = gf.getGabor(roi_gray, filters)
    hists = []
    for i in range(len(res)):
        # temp = np.zeros((100,100),np.uint8)
        # temp = res.copy()
        # temp = cv2.cvtColor(res[i], cv2.COLOR_BGR2GRAY)
        temp = res[i]
        hist = desc.describe(temp)
        hists.append(hist)
    pca = PCA(n_components=2)
    

model = svm.SVC(C=0.8, kernel = 'rbf', gamma = 20, decision_function_shape='ovr')


plt.show()

