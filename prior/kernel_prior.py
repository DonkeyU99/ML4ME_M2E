import numpy as np
import scipy
from scipy.stats import multivariate_normal, norm 

from collections import defaultdict
from sklearn.mixture import GaussianMixture

class Kernel_Prior():
    def __init__(self,kernel_size,prior_type='uniform'):
        self.kernel_size = kernel_size
        self.prior_type = prior_type
        self.params = None
    
    def pre_compute_params(self,kernel,**kwargs):
        params = defaultdict()
        if(self.prior_type=="uniform"):
            params['uniform_density'] = 1/(self.kernel_size[0]*self.kernel_size[1])     
        
        if (self.prior_type == "gaussian"):
            params['mean'] = np.mean(kernel)
            params['variance'] = np.var(kernel)
        
        if (self.prior_type == "gm"):
            assert 'components' in kwargs.keys()
            gm = GaussianMixture(n_components= kwargs['components'], random_state=0).fit(kernel)
            params['means'] = gm.means_
            params['covariances'] = gm.covariances_
            params['weights'] = gm.weights_
        
        self.params = params

    def predict_prob_K(self,inf_kernel,tau = 1):
        '''
        input: params = output of prior_k , input 
        output: log pdf of kernel !! kernel output needed? -> probability 만 필요할듯
        '''
        assert self.params != None, "Need to pre-compute params"

        if (self.prior_type == "uniform"):
            log_proba = np.sum(self.params['uniform_density'] * (-tau) * inf_kernel)

        if (self.prior_type == "gaussian"):
            log_proba = 0
            for i in range(inf_kernel.shape[0]):
                for j in range(inf_kernel.shape[1]):
                    log_proba -= norm.pdf(inf_kernel[i, j], self.params['mean'], self.params['variance']) * tau
        if (self.prior_type == "gm"):
            log_proba = 0
            for i in range(inf_kernel.shape[0]):
                for j in range(inf_kernel.shape[1]):
                    log_proba -= multivariate_normal.pdf(inf_kernel[i, j], self.params['means'], self.params['covariances']) * tau

        return log_proba
       
def kernel_estimate_K1_conv_K2(log_proba_1, log_proba_2):
    '''
    input: log_probability of kernel 1 & 2
    output: joint log probability of two independent distributions
    '''
    return log_proba_1 + log_proba_2