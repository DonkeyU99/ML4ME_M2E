import numpy as np
import cv2
import scipy
from scipy.stats import multivariate_normal, norm 

from optimization import optimizer
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
    def kernel_estimate_K1(params, kernel, kernel_size, prior_type, tau = 1):
        '''
        input: params = output of prior_k , input kernel
        output: log pdf of kernel !! kernel output needed? -> probability 만 필요할듯
        '''

        if (prior_type == "uniform"):
            log_proba = np.sum(params['uniform_density'] * (-tau) * kernel)
        
        if (prior_type == "gaussian"):
            log_proba = 0
            for i in kernel.shape[0]:
                for j in kernel.shape[1]:
                    log_proba -= norm.pdf(kernel[i, j], params['mean'], params['variance']) * tau
        if (prior_type == "gaussian_mixture"):
            log_proba = 0
            for i in kernel.shape[0]:
                for j in kernel.shape[1]:
                    log_proba -= multivariate_normal.pdf(kernel[i, j], params['means'], params['covariances']) * tau
            
        return log_proba
        
    def kernel_estimate_K2(params, kernel, kernel_size, prior_type, tau = 1):
        '''
        input: params = output of prior_k , input kernel
        output: log pdf of kernel !! kernel output needed? -> probability 만 필요할듯
        '''

        if (prior_type == "uniform"):
            log_proba = np.sum(params['uniform_density'] * (-tau) * kernel)
        
        if (prior_type == "gaussian"):
            log_proba = 0
            for i in kernel.shape[0]:
                for j in kernel.shape[1]:
                    log_proba -= norm.pdf(kernel[i, j], params['mean'], params['variance']) * tau
        if (prior_type == "gaussian_mixture"):
            log_proba = 0
            for i in kernel.shape[0]:
                for j in kernel.shape[1]:
                    log_proba -= multivariate_normal.pdf(kernel[i, j], params['means'], params['covariances']) * tau
            
        return log_proba
    ##
    def kernel_estimate_K1_conv_K2(log_proba_1, log_proba_2):
        '''
        input: log_probability of kernel 1 & 2
        output: joint log probability of two independent distributions
        '''
        return log_proba_1 + log_proba_2

    def load_dataset():
        pass
    
    ####Fill in distributions 
    def prior_K1(kernel_size, kernel, prior_type='uniform', n_components = 2):
        '''
        input: kernel_size: int, prior_type: "distribution string"
        output: parameters according to prior_type: type dictionary
        '''
        params = defaultdict()
        if(prior_type=="uniform"):
            params['uniform_density'] = 1/(kernel_size[0]*kernel_size[1])
        
        if (prior_type == "gaussian"):
            params['mean'] = np.mean(kernel)
            params['variance'] = np.var(kernel)
        
        if (prior_type == "gaussian_mixture"):
            
            gm = GaussianMixture(n_components= n_components, random_state=0).fit(kernel)

            params['means'] = gm.means_
            params['covariances'] = gm.covariances_
            params['weights'] = gm.weights_

        return params, kernel_size, prior_type
    
    def prior_K2(kernel_size, kernel, prior_type='uniform', n_components = 2):
        '''
        input: kernel_size: int, prior_type: "distribution string"
        output: parameters according to prior_type: type dictionary
        '''
        params = defaultdict()
        if(prior_type=="uniform"):
            params['uniform_density'] = 1/(kernel_size[0]*kernel_size[1])
        
        if (prior_type == "gaussian"):
            params['mean'] = np.mean(kernel)
            params['variance'] = np.var(kernel)
        
        if (prior_type == "gaussian_mixture"):
            
            gm = GaussianMixture(n_components= n_components, random_state=0).fit(kernel)

            params['means'] = gm.means_
            params['covariances'] = gm.covariances_
            params['weights'] = gm.weights_

        return params, kernel_size, prior_type
       
    # def prior_K1_conv_K2():
    #     ###Assumed K1 and K2 independent
    #     ###Computes the joint distribution of K1 conv K2
    #     pass
    
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
        '''
        Hyperparameters
        usually 1/(ζ**2 * τ) = 50 
        '''
        zeta_0 = None #TODO
        tau = None #TODO
        gamma = None #TODO
        
        prior_update(I,B,F1,F2) # -> HOW?
        # optimizer
        deft_optimizer = optimizer(likelihood = self.likelihood,zeta_0,gamma,tau,sigma,image,kernel)
        self.likelihood = deft_optimizer.likelihood_update()
        self.kernel = deft_optimizaer.kernel_update()
        
        return        

    def prior_update():
        prior_K1()
        prior_K2()
        prior_K1_conv_K2()
        prior_global()
        return prior1,2,3,4