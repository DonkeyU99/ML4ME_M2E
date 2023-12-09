import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim 

image_pathA = 'fig_save/origin.jpg' 
image_pathB = 'fig_save/33.jpg'  
imageA = cv2.imread(image_pathA)
imageB = cv2.imread(image_pathB)

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")

print("SSIM: {}".format(score))