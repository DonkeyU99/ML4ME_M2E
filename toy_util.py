import numpy as np

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
    
    def likelihood_blur():
    
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
        likelihood_blur()
        return likelihood_disrtribution
