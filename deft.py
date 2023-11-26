import numpy as np
from loader.likelihood_update import likelihood_update

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

    
    def prior_local():
    pass
    
    def likelihood_blur(noise,sigma=1.):
      grad_x = cv2.Sobel(noise,src=-1,dx=1,dy=0)
      grad_xx =cv2.Sobel(noise,src=-1,dx=2,dy=0)
      grad_y = cv2.Sobel(noise,src=-1,dx=0,dy=1)
      grad_yy =cv2.Sobel(noise,src=-1,dx=0,dy=2)
      grad_xy =cv2.Sobel(noise,src=-1,dx=1,dy=1)

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
        zeta_0 = ?? #TODO
        gamma = ?? #TODO
        # usually 1/(ζ**2 * τ) = 50 
        self.likelihood = likelihood_update(self.likelihood,zeta_0,gamma).update()
        return likelihood_disrtribution
