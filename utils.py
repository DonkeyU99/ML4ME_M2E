import cv2

def compute_gradient(source, type='x'):
    if(type == 'x'):
        return cv2.Sobel(source,src=-1,dx=1,dy=0)
    if(type == 'xx'):
        return cv2.Sobel(source,src=-1,dx=2,dy=0)
    if(type == 'y'):
        return cv2.Sobel(source,src=-1,dx=0,dy=1)
    if(type == 'yy'):
        return cv2.Sobel(source,src=-1,dx=0,dy=2)
    if(type == 'xy'):
        return cv2.Sobel(source,src=-1,dx=1,dy=1)