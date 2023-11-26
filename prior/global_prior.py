from utils import compute_gradient, Phi_func
import numpy as np
import cv2

class Prior_G_Fixed():
    def __init__(self, threshold, k=2.7, a=6.1e-4, b= 5):
        self.threshold = threshold
        self.k = k
        self.a = a
        self.b = b
    
    def compute_prior(self,img):
        grad_x = compute_gradient(img, 'x')
        grad_y = compute_gradient(img, 'y')
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        prob = Phi_func(img, self.threshold, self.k, self.a,self.b)
        return np.sum(prob)