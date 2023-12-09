import numpy as np
import scipy
from utils import compute_gradient
from scipy.stats import multivariate_normal
import cv2
import matplotlib.pyplot as plt

def smooth_region(img, kernel_size, threshold,cnt):
    local_std = np.zeros(img.shape)
    std_kernel = np.ones((kernel_size, kernel_size))
    im = np.array(img, dtype = float)
    im2 = im**2
    ones = np.ones((im.shape[0], im.shape[1]))

    s = scipy.signal.convolve2d(im, std_kernel, mode="same")
    s2 = scipy.signal.convolve2d(im2, std_kernel, mode="same")
    ns = scipy.signal.convolve2d(ones, std_kernel, mode="same")
    local_std = np.sqrt((s2 - s**2 / ns) / ns)

    show_img = local_std
    show_img[show_img > threshold[0]] = 0
    plt.imshow(show_img, cmap='gray')
    plt.savefig(f"./smooth_reg/fig{cnt}.jpg")
        
    region = local_std < threshold[0]

    return region # True면 Omega에 들어가는 pixel (img.shape[1], img.shape[2])

def prior_local(latent_img, img, kernel_size, sigma_1, threshold = np.array([5, 5, 5])):
    region = smooth_region(img, kernel_size, threshold)
    idx = np.where(region == True)

    delta_grad_x = compute_gradient(latent_img,'x') - compute_gradient(img, 'x')
    delta_grad_y = compute_gradient(latent_img, 'y') - compute_gradient(img, 'y')

    px = multivariate_normal.logpdf(delta_grad_x[idx].flatten(), mean=0, cov=sigma_1)
    py = multivariate_normal.logpdf(delta_grad_y[idx].flatten(), mean=0, cov=sigma_1)

    log_likelihood = np.sum(px) + np.sum(py)

    return log_likelihood