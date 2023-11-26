import numpy as np
import cv2
import scipy
from scipy.stats import multivariate_normal, norm 

from optimizer import optimizer
from utils import compute_gradient
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from prior.kernel_prior import Kernel_Prior
from prior.local_prior import Local_Prior
from prior.likelihood_blur import Blur_Likelihood
from prior.global_prior import Prior_G_Fixed

class DeFT():
    def __init__(self,k_size=(3,3)):
        self.prior_K1 = Kernel_Prior(k_size,'uniform')
        self.prior_K2= Kernel_Prior(k_size,'uniform')
        self.local_prior = Local_Prior(k_size,'RGB')
        self.global_prior = Prior_G_Fixed()
        self.likelihood = Blur_Likelihood()

    ##def bayesian_update_K1():
    ##    pass
    ##
    def load_dataset():
        pass
    
    def prior_global(img_size,prior_type='uniform'):
        if(prior_type=="uniform"):
            ###Img size given as 2D
            return np.full(img_size,1/(img_size[0]*img_size[1]))

    # def std_convoluted(image, N):
    #     im = np.array(image, dtype=float)
    #     im2 = im**2
    #     ones = np.ones(im.shape)
    
    #     kernel = np.ones((2*N+1, 2*N+1))
    #     s = scipy.signal.convolve2d(im, kernel, mode="same")
    #     s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
    #     ns = scipy.signal.convolve2d(ones, kernel, mode="same")
    
    #     return np.sqrt((s2 - s**2 / ns) / ns)


    
    ##def bayesian_inference():
    ##    pass
    
    def train(I,B,F1,F2):
        prior_update(I,B,F1,F2) # -> HOW?
        # optimizer
        self.L, self.kernel = optimizer.optimize(self.kernel, self.image, sigma, max_iterations=100)
        return

    def prior_update():
        prior_K1()
        prior_K2()
        prior_K1_conv_K2()
        prior_global()
        return prior1,2,3,4