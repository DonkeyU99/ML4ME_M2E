import numpy as np
from loader.likelihood_update import likelihood_update
from scipy.stats import multivariate_normal
from utils import compute_gradient
import cv2
import scipy

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
    ##def kernel_estimate_K1():
    ##    pass
    ##
    ##def kernel_estimate_K2():
    ##    pass
    ##
    ##def kernel_estimate_K1_conv_K2():
    ##    pass

    def load_dataset():
        pass
    
    ####Fill in distributions 
    def prior_K1(kernel_size,prior_type='uniform'):
        if(prior_type=="uniform"):
            return np.full(kernel_size,1/(kernel_size[0]*kernel_size[1]))
    
    def prior_K2(kernel_size,prior_type='uniform'):
        if(prior_type=="uniform"):
            return np.full(kernel_size,1/(kernel_size[0]*kernel_size[1]))
   
    def prior_K1_conv_K2():
        ###Assumed K1 and K2 independent
        ###Computes the joint distribution of K1 conv K2
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

    def smooth_region(img, kernel_size, threshold):
        local_std = np.zeros(img.shape)
        std_kernel = np.ones((kernel_size, kernel_size))
        im = np.array(img, dtype = float)
        im2 = im**2
        ones = np.ones((im.shape[1], im.shape[2]))

        for i in range(3):
            s = scipy.signal.convolve2d(im[i], std_kernel, mode="same")
            s2 = scipy.signal.convolve2d(im2[i], std_kernel, mode="same")
            ns = scipy.signal.convolve2d(ones, std_kernel, mode="same")
            local_std[i] = np.sqrt((s2 - s**2 / ns) / ns)
        
        region = (local_std[0] > threshold[0]) & (local_std[1] > threshold[1]) & (local_std[2] > threshold[2])

        return region # True면 Omega에 들어가는 pixel (img.shape[1], img.shape[2])

    def prior_local(latent_img, img, kernel_size, sigma, threshold = np.array([5, 5, 5])):
        smooth_region = smooth_region(img)
                
    def likelihood_blur(noise,sigma=1.):
      grad_x = compute_gradient(noise,src=-1,dx=1,dy=0)
      grad_xx = compute_gradient(noise,src=-1,dx=2,dy=0)
      grad_y = compute_gradient(noise,src=-1,dx=0,dy=1)
      grad_yy = compute_gradient(noise,src=-1,dx=0,dy=2)
      grad_xy = compute_gradient(noise,src=-1,dx=1,dy=1)

      sigma_1 = np.sqrt(2)*sigma
      sigma_2 = 2*sigma

      log_likelihood = 0
      for i in (grad_x,grad_y):
          log_likelihood += np.sum(multivariate_normal.logpdf(i, mean=0, cov=sigma_1))  #allow_singular=False
              
      for i in (grad_xx,grad_xy,grad_yy):
          log_likelihood += np.sum(multivariate_normal.logpdf(i, mean=0, cov=sigma_2))
          
      return log_likelihood
    
    ##def bayesian_inference():
    ##    pass
    
    def train(I,B,F1,F2):
        ##optimizer
        prior_update(I,B,F1,F2)
        likelihood_update(I,B,F1,F2)
    
        pass
    
    def prior_update():
        prior_K1()
        prior_K2()
        prior_K1_conv_K2()
        prior_global()
        return prior1,2,3,4
    
    def likelihood_update():
        zeta_0 = None #TODO
        tau = None #TODO
        gamma = None #TODO
        # usually 1/(ζ**2 * τ) = 50 
        self.likelihood = likelihood_update(likelihood = self.likelihood,zeta_0,gamma,tau,sigma,image,kernel).update()
        return likelihood_disrtribution
