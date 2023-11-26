import numpy as np
from collections import defaultdict
from scipy.stats import norm, multivariate_normal
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
            log_proba = np.sum(- tau * np.full(kernel_size * kernel_size, params['uniform_density']))
        
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
            log_proba = np.sum(- tau * np.full(kernel_size * kernel_size, params['uniform_density']))
        
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

    
    def prior_local():
        pass
    
    def likelihood_blur():
        pass
    
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
