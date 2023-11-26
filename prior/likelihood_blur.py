from utils import compute_gradient
import numpy as np
from scipy.stats import multivariate_normal, norm 

class Blur_Likelihood():
    def __init__(self,sigma=1.):
        self.sigma = sigma

    def compute_likelihood(self,noise):
        sigma_1 = np.sqrt(2)*self.sigma
        sigma_2 = 2*self.sigma

        log_likelihood = 0
        for i in ('x','y'):
            log_likelihood += np.sum(multivariate_normal.logpdf(compute_gradient(noise,i).flatten(), mean=0, cov=sigma_1))  #allow_singular=False
              
        for i in ('xx','xy','yy'):
            log_likelihood += np.sum(multivariate_normal.logpdf(compute_gradient(noise,i).flatten(), mean=0, cov=sigma_2))     
        return log_likelihood