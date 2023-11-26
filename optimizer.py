import numpy as np
import numpy.fft as fft
from utils import compute_gradient, psf2otf
import l1ls as L

class Optimizer():
  def __init__(self, kernel, image, sigma, max_iterations = 15):
    self.I = image
    
    self.f = kernel
    self.L = None
    self.Psi = None
    
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
  
    # self.lambda_1 = 1/tau
    # self.lambda_2 = 1/(sigma**2*tau)
    
  def omega(self, q):
    return 1/((self.zeta_0**2)*self.tau*(2**q))
    
  def conj_fft(self, array): # conj(FFT) operator
    return np.conj(np.fft.fft(array))
  

  def gradient_filter(self, type):
    #TODO
    
  def update_Psi(self):
    # TODO
    
  def update_L(self):
    gradients = ['x','y','xx','xy','yy']
    gradient_filters = self.gradient_filter(gradients)
    self.Psi = [compute_gradient(self.L, type='x'),compute_gradient(self.L, type='y')]
    
    grad_x = self.gradient_filter('x')
    grad_y = self.gradient_filter('y')

    Delta = np.sum(self.omega(len(gradients))*self.conj_fft(gradient_filters)*gradient_filters)     # q=1 for x y, q=2 for xx xy yy

    numer = np.sum(self.conj_fft(self.f)*fft(self.I)*Delta + self.gamma*self.conj_fft(grad_x)*fft(self.Psi[0]) + self.gamma*self.conj_fft(grad_y)*fft(self.Psi[1]) )
    denom = np.sum(self.conj_fft(self.f)*fft(self.f)*Delta + self.gamma*self.conj_fft(grad_x)*fft(grad_x) + self.gamma*self.conj_fft(grad_y)*fft(grad_y))
    new_L = np.fft.ifft(numer/denom)
    
    self.delta_L = new_L - self.L
    self.L = new_L
  
  def update_f(self):  
    A =
    B = 
    [new_f, status, hist] = L.l1ls(A, y=B, lmbda=1)

    self.delta_f = new_f - self.f
    self.f = new_f    
  
  def optimize(self):
    iteration = 0
    while iteration < self.max_iterations:
      # Inner loop to optimize L
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
        iteration += 1
        self.gamma *= 2
    # Return L and f after optimization
    return self.L, self.f