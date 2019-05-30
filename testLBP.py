import numpy as np
import matplotlib.pyplot as plt
import cv2
from localBinaryPatterns import LocalBinaryPatterns

imagePath = 'C:/Users/User/Documents/test/1.jpg'
image = cv2.imread(imagePath)
desc = LocalBinaryPatterns(numPoints=24,radius=8)
desc.describe(image)
