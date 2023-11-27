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

def toeplitz_matrix(input, kernel_size):
    height, width, channel = input.shape
    p = int((kernel_size - 1) / 2)
    result = np.zeros((channel, height*width, kernel_size**2))
    for k in range(channel):
        input_c = np.pad(input[:, :, k], ((p,p),(p,p)), 'constant', constant_values=0)
        for i in range(height):
            for j in range(width):
                local_mat = input_c[i:i+kernel_size, j:j+kernel_size]
                local_mat = local_mat.flatten()
                result[k, i*width + j,:] = local_mat

    return result # (3, height, width) 느낌으로 출력