from utils import compute_gradient
import numpy as np
import cv2

def prior_global(img, threshold, k = 2.7, a = 6.1e-4, b = 5):
    grad_x = compute_gradient(img, 'x')
    grad_y = compute_gradient(img, 'y')

    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    prob = np.where(grad_magnitude <= threshold, -k*grad_magnitude, -b-a*grad_magnitude**2)
    return np.sum(prob)
