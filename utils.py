import cv2

def compute_gradient(source, src=-1, dx=1,dy=0):
    return cv2.Sobel(source,src=-1,dx=dx,dy=dy)