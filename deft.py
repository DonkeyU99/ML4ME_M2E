import numpy as np
import cv2
import scipy
from scipy.stats import multivariate_normal, norm 

from optimizer import optimizer
from utils import compute_gradient
from collections import defaultdict
from sklearn.mixture import GaussianMixture

class DeFT():
    def __init__(self):
        self.prior1 = None
        self.prior2= None
        self.prior3 = None
        self.prior4 = None
        self.likelihood = None

        self.kernel_sizes = 2

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

    def likelihood_blur(noise,sigma=1.):

      sigma_1 = np.sqrt(2)*sigma
      sigma_2 = 2*sigma

      log_likelihood = 0
      for i in ('x','y'):
          log_likelihood += np.sum(multivariate_normal.logpdf(compute_gradient(noise,i), mean=0, cov=sigma_1))  #allow_singular=False
              
      for i in ('xx','xy','yy'):
          log_likelihood += np.sum(multivariate_normal.logpdf(compute_gradient(noise,i), mean=0, cov=sigma_2))
          
      return log_likelihood
    
    ##def bayesian_inference():
    ##    pass
    
    def train(I,B,F1,F2):
        prior_update(I,B,F1,F2) # -> HOW?
        # optimizer
        self.likelihood, self.kernel = optimizer.optimize(self.likelihood,self.image,sigma,max_iterations=100)
        return

    def prior_update():
        prior_K1()
        prior_K2()
        prior_K1_conv_K2()
        prior_global()
        return prior1,2,3,4