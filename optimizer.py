import numpy as np
import numpy.fft as fft
from utils import compute_gradient, Phi_func
import l1ls as L
from scipy.optimize import minimize
from local_prior import smooth_region
from scipy.linalg import convolution_matrix

class Optimizer():
    def __init__(self, image, kernel_size, sigma, max_iterations = 15):
        self.I = image
        self.I_grad_x = compute_gradient(image, 'x')
        self.I_grad_y = compute_gradient(image, 'y')
        self.kernel_size = kernel_size

        self.f = np.diag(np.full(kernel_size, 1))
        # initialize L, psi_X, psi_y
        self.L = image
        self.Psi_x = compute_gradient(self.L, 'x')
        self.Psi_y = compute_gradient(self.L, 'y')
        
        '''
        Hyperparameters
        usually 1/(ζ**2 * τ) = 50  
        '''
        self.delta_f = None
        self.delta_L = None
        self.delta_Psi = None
        
        self.zeta_0 = None #TODO
        self.tau = None #TODO
        self.gamma = 2 #TODO
      
        self.lambda_1 = 1/self.tau
        self.lambda_2 = 1/(sigma**2*self.tau)

        self.max_iterations = 15

    def omega(self, input):
        if input == '0':
            q = 0
        elif input == 'x' or input == 'y':
            q = 1
        elif input == 'xx' or input == 'yy' or input == 'xy':
            q = 2
        return 1/((self.zeta_0**2)*self.tau*(2**q))

    def conj_fft(self, array): # conj(FFT) operator
        return np.conj(np.fft.fft(array))

    def gradient_filter(self, type = "x"):
        if (type == "x"):
            filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            

        if (type == "y"):
            filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

        if (type == "xy"):
            filter = [[1, 0, -1], [0, 0, 0], [-1, 0, 1]]

        if (type == "xx"):
            filter = [[1, -2, 1], [2, -4, 2], [1, -2, 1]]

        if (type == "yy"):
            filter = [[1, 2, 1], [-2, -4, -2], [1, 2, 1]]
        
        return filter 

    
    def update_Psi(self):
        mask = 1*smooth_region(self.L, self.kernel_size, self.threshold_smooth_region)
        for i in range(self.I.shape[0]):
            for j in range(self.I.shape[1]):
                for k in range(3):
                    fun_x = lambda x: self.lambda_1*np.abs(Phi_func(x, self.threshold_Phi_func))+self.lambda_2*mask[i, j]*(x-self.I_grad_x[i, j, k])**2+self.gamma*(x-compute_gradient(self.L, 'x')[i, j, k])**2
                    fun_y = lambda x: self.lambda_1*np.abs(Phi_func(x, self.threshold_Phi_func))+self.lambda_2*mask[i, j]*(x-self.I_grad_y[i, j, k])**2+self.gamma*(x-compute_gradient(self.L, 'y')[i, j, k])**2
                    objective_x = lambda x: fun_x(x)
                    objective_y = lambda x: fun_y(x)
                   
                    self.Psi_x[i, j, k] = minimize(objective_x, self.Psi_x[i, j, k]).x[0]
                    self.Psi_y[i, j, k] = minimize(objective_y, self.Psi_y[i, j, k]).x[0]

    def update_L(self):
        gradients = ['0', 'x','y','xx','xy','yy']
        gradient_filters = self.gradient_filter(gradients)

        grad_x = self.gradient_filter('x')
        grad_y = self.gradient_filter('y')

        Delta = np.sum([self.omega(grad) for grad in gradients]*self.conj_fft(gradient_filters)*fft(gradient_filters))     # q=1 for x y, q=2 for xx xy yy

        numer = np.sum(self.conj_fft(self.f)*fft(self.I)*Delta + self.gamma*self.conj_fft(grad_x)*fft(self.Psi_x) + self.gamma*self.conj_fft(grad_y)*fft(self.Psi_y))
        denom = np.sum(self.conj_fft(self.f)*fft(self.f)*Delta + self.gamma*self.conj_fft(grad_x)*fft(grad_x) + self.gamma*self.conj_fft(grad_y)*fft(grad_y))
        new_L = np.fft.ifft(numer/denom)
        
        self.delta_L = new_L - self.L
        self.L = new_L
    
    def update_f(self):
        gradients = ['0', 'x','y','xx','xy','yy']
        for grad in gradients:
            convolution_matrix()
            몰라 씨발

    def update_f(self):  
        gradients = ['x','y','xx','xy','yy']
        A = np.array([])
        for var in gradients:
            C = circulant(f[0])
            C = np.kron(C, circulant(f[:, 0]))
            
            # Use view_as_windows to create overlapping patches of the input image
            patches = view_as_windows(compute_gradient(self.L,type=var), (len(self.f), len(self.f)), step=1).reshape(-1, len(self.f)**2)
            
            A += self.omega(var)*patches
            
        B = np.sum([self.omega(var)*compute_gradient(self.I,type=var) for var in gradients])
        [new_f, status, hist] = L.l1ls(A, y=B, lmbda=1)

        self.delta_f = new_f - self.f
        self.f = new_f    
    
    def optimize(self):
        iteration = 0
        # Inner loop to optimize L
        while iteration < self.max_iterations:
            while True:
                # Update Ψ and compute L
                self.update_Psi()
                self.update_L()
                if  np.linalg.norm(self.delta_L) < 1e-5 and np.linalg.norm(self.delta_Psi) < 1e-5:
                    break
            # Update f
            self.update_f()
            if np.linalg.norm(self.delta_f) < 1e-5:
                break
            self.gamma *= 2
            iteration += 1
        # Return L and f after optimization
        return self.L, self.f