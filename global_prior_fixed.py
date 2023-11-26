from utils import compute_gradient, Phi_func
import numpy as np
import cv2

def prior_global(img, threshold, k = 2.7, a = 6.1e-4, b = 5):
    grad_x = compute_gradient(img, 'x')
    grad_y = compute_gradient(img, 'y')

    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    prob = Phi_func(img, threshold, k, a, b)
    return np.sum(prob)