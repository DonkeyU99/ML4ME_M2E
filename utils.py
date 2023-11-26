import cv2
import numpy as np

def compute_gradient(source, type='x', ddepth = cv2.CV_64F):
    if(type == 'x'):
        return cv2.Sobel(source,ddepth,dx=1,dy=0)
    if(type == 'xx'):
        return cv2.Sobel(source,ddepth,dx=2,dy=0)
    if(type == 'y'):
        return cv2.Sobel(source,ddepth,dx=0,dy=1)
    if(type == 'yy'):
        return cv2.Sobel(source,ddepth,dx=0,dy=2)
    if(type == 'xy'):
        return cv2.Sobel(source,ddepth,dx=1,dy=1)
      
def Phi_func(x, threshold, k = 2.7, a = 6.1e-4, b = 5):
    return np.where(x <= threshold, -k*x, -b-a*x**2)