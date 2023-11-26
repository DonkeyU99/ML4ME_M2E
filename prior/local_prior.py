import numpy as np
import scipy
from utils import compute_gradient
from scipy.stats import multivariate_normal
import cv2


class Local_Prior():
    def __init__(self,kernel_size,type='RGB',**kwargs):
        self.sigma1 = None
        self.kernel_size = kernel_size
        assert type in {'RGB'}
        self.type = type
        if(type =='RGB'):
            if 'threshold' in kwargs.keys():
                self.threshold = kwargs['threshold']
            else:
                self.threshold = np.array([5, 5, 5])

    def smooth_region(self, img):
        if(self.type == 'RGB'):
            local_std = np.zeros(img.shape)
            std_kernel = np.ones(self.kernel_size)
            im = np.array(img, dtype = float)
            im2 = im**2
            ones = np.ones((im.shape[0], im.shape[1]))

            for i in range(3):
                s = scipy.signal.convolve2d(im[:,:,i], std_kernel, mode="same")
                s2 = scipy.signal.convolve2d(im2[:,:,i], std_kernel, mode="same")
                ns = scipy.signal.convolve2d(ones, std_kernel, mode="same")
                local_std[:,:,i] = np.sqrt((s2 - s**2 / ns) / ns)

            region = (local_std[:,:,0] < self.threshold[0]) & (local_std[:,:,1] < self.threshold[1]) &  (local_std[:,:,2] < self.threshold[2])

            return region # True면 Omega에 들어가는 pixel (img.shape[1], img.shape[2])

    def prior_local(self,latent_img, img, sigma_1):
        if(self.type == 'RGB'):
            self.sigma_1 = sigma_1
            region = self.smooth_region(img)
            idx = np.where(region == True)

            delta_grad_x = compute_gradient(latent_img,'x') - compute_gradient(img, 'x')
            delta_grad_y = compute_gradient(latent_img, 'y') - compute_gradient(img, 'y')

            px = multivariate_normal.logpdf(delta_grad_x[idx].flatten(), mean=0, cov=self.sigma_1)
            py = multivariate_normal.logpdf(delta_grad_y[idx].flatten(), mean=0, cov=self.sigma_1)

            log_likelihood = np.sum(px) + np.sum(py)

            return log_likelihood