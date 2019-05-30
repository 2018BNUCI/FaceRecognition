import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('C:/Users/User/Documents/test/1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
face_cascade = cv2.CascadeClassifier('C:/Users/User/Documents/test/opencv/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/User/Documents/test/opencv/haarcascade_eye.xml')

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)


for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h,x:x+w]
    roi = image[y:y+h,x:x+w]
    img_face = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    eyes = eye_cascade.detectMultiScale(roi_gray,1.03,5,0,(40,40))
    # print(len(eyes))
    flag = int(0)
    for (ex,ey,ew,eh) in eyes:
        img = cv2.rectangle(roi, (ex,ey), (ex+ew,ey+eh), (255,0,0),2 )
        flag+=1
        if flag == 4:
            break

mask = np.zeros(image.shape[:2],np.uint8)

bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

rect = tuple(faces[0])
print(rect)

cv2.grabCut(image,mask,rect,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
image = image*mask2[:,:,np.newaxis]

plt.imshow(image),plt.xticks([]),plt.yticks([]),plt.title('after grabCut')
plt.colorbar()

# plt.subplot(1,3,1)
# plt.imshow(img_face)
# plt.title("face"),plt.xticks([]),plt.yticks([])
#
# plt.subplot(1,3,2)
# plt.imshow(img),plt.title("head and eyes"),plt.xticks([]),plt.yticks([])
#
# plt.subplot(1,3,3)
# plt.imshow(roi),plt.title("head"),plt.xticks([]),plt.yticks([])

plt.show()
