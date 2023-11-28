import numpy as np
import numpy.fft as fft
from utils import compute_gradient, Phi_func, toeplitz_matrix
from scipy.optimize import minimize
from local_prior import smooth_region
import cv2

class Optimizer():
    def __init__(self, image, kernel_size, sigma = None, max_iterations = 15):
        self.I = image
        self.I_flat = self.I.flatten()
        self.I_grad_x = compute_gradient(image, 'x')
        self.I_grad_y = compute_gradient(image, 'y')
        self.I_grad_x_flat = self.I_grad_x.flatten()
        self.I_grad_y_flat = self.I_grad_y.flatten()
        self.I_grad_xx_flat = compute_gradient(image, 'xx').flatten()
        self.I_grad_xy_flat = compute_gradient(image, 'xy').flatten()
        self.I_grad_yy_flat = compute_gradient(image, 'yy').flatten()
        self.kernel_size = kernel_size

        self.height = image.shape[0] + kernel_size - 1
        self.width = image.shape[1] + kernel_size - 1
        self.F_I = fft.fft2(self.I,(self.height, self.width), axes = (0, 1))
        

        
        # initialize L, psi_X, psi_y
        self.L = image
        self.Psi_x = compute_gradient(self.L, 'x')
        self.Psi_y = compute_gradient(self.L, 'y')
        self.f = np.diag(np.full(kernel_size, 1))
        self.f_flat = self.f.flatten()
        
        '''
        Hyperparameters
        usually 1/(ζ**2 * τ) = 50  
        '''
        self.delta_f = None
        self.delta_L = None
        self.delta_Psi_x = None
        self.delta_Psi_y = None
        
        self.zeta_0 = 1 #TODO
        # tau : 2 ~ 500
        self.tau = 0.25 #TODO
        self.gamma = 2 #TODO

        # 0.002 ~ 0.5 / 10 ~ 25
        # self.lambda_1 = 1/self.tau
        # self.lambda_2 = 1/(sigma**2*self.tau)

        self.lambda_1 = 0.05
        self.lambda_2 = 15

        self.k1 = 1.3
        self.k2 = 1.5

        self.weight = np.array([self.omega(grad) for grad in ['0','x','y','xx','xy','yy']])
        self.sigma_star = np.array([fft.fft2(self.gradient_filter(grad),(self.height, self.width)) for grad in ['0','x','y','xx','xy','yy']])
        self.delta = np.einsum('i,ijk->jk', self.weight, np.einsum('ijk,ijk->ijk', self.sigma_star, np.conjugate(self.sigma_star)))

        self.max_iterations = max_iterations
        self.threshold_smooth_region = np.array([5, 5, 5])
        self.threshold_Phi_func = 5
        
    def gradient_filter(self, type):
        if type == '0':
            filter = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]

        if type == 'x':
            filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            
        if type == 'y':
            filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

        if type == 'xy':
            filter = [[1, 0, -1], [0, 0, 0], [-1, 0, 1]]

        if type == 'xx':
            filter = [[1, -2, 1], [2, -4, 2], [1, -2, 1]]

        if type == 'yy':
            filter = [[1, 2, 1], [-2, -4, -2], [1, 2, 1]]
        return np.array(filter)

    def omega(self, input):
        if input == '0':
            q = 0
        elif input == 'x' or input == 'y':
            q = 1
        elif input == 'xx' or input == 'yy' or input == 'xy':
            q = 2
        return 50/2**q

    def conj_fftn(self, array, axes=(0, 1)): # conj(FFT) operator
        return np.conj(np.fft.fftn(array,(self.height, self.width), axes=axes))
    
    def update_Psi(self):
        mask = 1*smooth_region(self.L, self.kernel_size, self.threshold_smooth_region)
        new_Psi_x = np.zeros_like(self.Psi_x)
        new_Psi_y = np.zeros_like(self.Psi_y)
        for i in range(self.I.shape[0]):
            for j in range(self.I.shape[1]):
                for k in range(3):
                    fun_x = lambda x: self.lambda_1*np.abs(Phi_func(x, self.threshold_Phi_func))+self.lambda_2*mask[i, j]*(x-self.I_grad_x[i, j, k])**2+self.gamma*(x-compute_gradient(self.L, 'x')[i, j, k])**2
                    fun_y = lambda x: self.lambda_1*np.abs(Phi_func(x, self.threshold_Phi_func))+self.lambda_2*mask[i, j]*(x-self.I_grad_y[i, j, k])**2+self.gamma*(x-compute_gradient(self.L, 'y')[i, j, k])**2
                    objective_x = lambda x: fun_x(x)
                    objective_y = lambda x: fun_y(x)

                    #print(f"Ratio : {i+1}/{self.I.shape[0]}, {j+1}/{self.I.shape[1]}, {k+1}/3")

                    new_Psi_x[i, j, k] = minimize(objective_x, self.Psi_x[i, j, k].item(),method="BFGS").x[0]
                    new_Psi_y[i, j, k] = minimize(objective_y, self.Psi_y[i, j, k].item(),method="BFGS").x[0]
        self.delta_Psi_x = new_Psi_x - self.Psi_x
        self.delta_Psi_y = new_Psi_y - self.Psi_y

        self.Psi_x = new_Psi_x
        self.Psi_y = new_Psi_y

    def update_L(self):
        
        print('start update L')
        F_psi_x = fft.fft2(self.Psi_x,(self.height, self.width), axes = (0, 1))
        F_psi_y = fft.fft2(self.Psi_y,(self.height, self.width), axes = (0, 1))

        F_f = fft.fft2(self.f,(self.height, self.width), axes = (0, 1)) # 3x3 -> (i+f-1)x(i+f-1)
        self.F_I #30x30x3(for img 28, kernel 3)
        self.delta #3x3
        
        denom = np.conjugate(F_f) * F_f * self.delta + self.gamma * (np.conjugate(self.sigma_star[1]) * self.sigma_star[1] + np.conjugate(self.sigma_star[2]) * self.sigma_star[2])
        new_L = np.zeros((self.height, self.width,3))
        for i in range(3):
            numer = np.conjugate(F_f) * self.F_I[:,:,i] * self.delta + self.gamma * (np.conjugate(self.sigma_star[1]) * F_psi_x[:,:,i] + np.conjugate(self.sigma_star[2]) * F_psi_y[:,:,i])
            
            new_L[:,:,i] = fft.ifft2(numer/denom,(self.height, self.width), axes = (0, 1))
        # image cropping
        st = (self.kernel_size - 1) // 2 - 1
        ed = self.I.shape[0]
        ed_ = self.I.shape[1] # if image is not square
        new_L = new_L[st:ed, st:ed_, :]

        self.delta_L = new_L - self.L
        self.L = new_L
        print('delta_L:',self.delta_L.shape)
        print('self.L:', self.L.shape)

        
        #Delta = np.sum([self.omega(grad) for grad in gradients]*self.conj_fft(gradient_filters)*fft#(gradient_filters))     # q=1 for x y, q=2 for xx xy yy

        """여기 고쳐야함 fft dimension
        Delta = np.sum([self.omega(grad)*self.conj_fftn(self.gradient_filter(grad),axes=(0, 1))*fft.fftn(self.gradient_filter(grad),axes=(0, 1)) for grad in gradients]) # q=1 for x y, q=2 for xx xy yy

        print(Delta)
        #np.array([self.f,self.f,self.f])
        numer = np.sum(self.conj_fftn(self.f,axes=(0, 1))*fft.fftn(self.I,axes=(0, 1))*Delta + self.gamma*self.conj_fftn(grad_x)*fft.fftn(self.Psi_x,axes=(0, 1)) + self.gamma*self.conj_fftn(grad_y,axes=(0, 1))*fft.fftn(self.Psi_y,axes=(0, 1)))
        denom = np.sum(self.conj_fftn(self.f,axes=(0, 1))*fft.fftn(self.f,axes=(0, 1))*Delta + self.gamma*self.conj_fftn(grad_x,axes=(0, 1))*fft.fftn(grad_x,axes=(0, 1)) + self.gamma*self.conj_fftn(grad_y,axes=(0, 1))*fft.fftn(grad_y,axes=(0, 1)))
        new_L = np.fft.ifftn(numer/denom,axes=(0, 1))
        
        self.delta_L = new_L - self.L
        self.L = new_L
        """
    
    def update_f(self):
        self.f_flat = self.f.flatten()
        A0 = toeplitz_matrix(self.L, self.kernel_size)
        Ax = toeplitz_matrix(compute_gradient(self.L, 'x'))
        Ay = toeplitz_matrix(compute_gradient(self.L, 'y'))
        Axx = toeplitz_matrix(compute_gradient(self.L, 'xx'))
        Axy = toeplitz_matrix(compute_gradient(self.L, 'xy'))
        Ayy = toeplitz_matrix(compute_gradient(self.L, 'yy'))
        B0 = self.I_flat
        Bx = self.I_grad_x_flat
        By = self.I_grad_y_flat
        Bxx = self.I_grad_xx_flat
        Bxy = self.I_grad_xy_flat
        Byy = self.I_grad_yy_flat

        objective_fun = lambda x: self.omega('0')*np.linalg.norm(A0@x-B0) + self.omega('x')*np.linalg.norm(Ax@x-Bx) + self.omega('y')*np.linalg.norm(Ay@x-By) + self.omega('xx')*np.linalg.norm(Axx@x-Bxx) + self.omega('0')*np.linalg.norm(Axy@x-Bxy) + self.omega('yy')*np.linalg.norm(Ayy@x-Byy) + np.sum(np.abs(x))
        initial_guess = self.f.flatten()

        self.f_flat = minimize(objective_fun, initial_guess, method='BFGS').x
        new_f = self.f_flat.reshape((self.kernel_size, self.kernel_size))
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
                print("------L delta-------")
                norm_L = np.linalg.norm(self.delta_L)
                print(norm_L)

                print("------Psi delta-------")
                norm_psi = np.linalg.norm(self.delta_Psi_x)+np.linalg.norm(self.delta_Psi_y)
                print(norm_psi)

                if norm_L < 1e-5 and norm_psi< 1e-5:
                    break
            # Update f
            self.update_f()
            print("--f delta--")
            norm_delta = np.linalg.norm(self.delta_f)
            print(norm_delta)
            if norm_delta < 1e-5:
                break
            self.gamma *= 2
            self.lambda_1 /= self.k1
            self.lambda_2 /= self.k2
            iteration += 1
        # Return L and f after optimization
        return self.L, self.f
    

#img = cv2.imread('data/toy_dataset/test.jpg')

img = np.random.randint(0,256,(28,28,3)).astype(float)

a = Optimizer(img, 3, max_iterations = 5)
a.optimize()
# update L & update psi 부분 확인


# L,f = a.optimize()
# print(L)