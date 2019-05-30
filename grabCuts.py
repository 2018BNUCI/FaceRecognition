import numpy as np
from cv2 import grabCut,GC_INIT_WITH_RECT

class grabcut:
    #这里的图片最好使用imageOri(RGB)
    def __init__(self, rect, count):
        self.rect = rect
        self.count = count
    def gcresult(self, image):
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        grabCut(image, mask, self.rect, bgdModel, fgdModel, self.count, GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        image = image * mask2[:, :, np.newaxis]

        return image