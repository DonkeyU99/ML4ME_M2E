import numpy
import numpy.fft.fft as fft
from utils import compute_gradient, psf2otf

class optimizer():
  def __init__(self, kernel, image, likelihood,sigma, max_iterations = 100):
    self.I = image
    
    self.f = kernal
    self.L = likelihood
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
    self.gamma = None #TODO
  
    # self.lambda_1 = 1/tau
    # self.lambda_2 = 1/(sigma**2*tau)
    
  def omega(q):
    return 1/((zeta_0**2)*tau*(2**q))
    
  def conj_fft(array): # conj(FFT) operator
    return np.conj(np.fft.fft(array))
  

  def gradient_filter(type):
    #TODO
    
  def update_Psi():
    # TODO
    
  def update_L():
    gradients = ['x','y','xx','xy','yy']
    gradient_filters = gradient_filter(gradients)
    self.Psi = [compute_gradient(likelihood, type='x'),compute_gradient(likelihood, type='y')]
    
    grad_x = gradient_filter('x')
    grad_y = gradient_filter('y')
    [Psi_x, Psi_y] = self.Psi

    Delta = np.sum(omega(len(gradients))*conj_fft(gradient_filters)*gradient_filters)     # q=1 for x y, q=2 for xx xy yy

    numer = np.sum( conj_fft(kernel)*fft(image)*Delta + gamma*conj_fft(grad_x)*fft(phi_x) + gamma*conj_fft(grad_y)*fft(phi_y) )
    numer = np.sum( conj_fft(kernel)*fft(kernel)*Delta + gamma*conj_fft(grad_x)*fft(grad_x) + gamma*conj_fft(grad_y)*fft(grad_y) )
    new_L = np.fft.ifft(numer/denom)
    
    self.delta_L = new_L - self.L
    self.L = new_L
  
  def update_f():  
    #TODO
    self.delta_f = new_f - self.f
    self.f = new_f    
  
  def optimize(self):
    iteration = 0
    while iteration < self.max_iterations:
      # Inner loop to optimize L
      while True:
        # Update Ψ and compute L
        update_Psi()
        update_L()
        if  np.linalg.norm(self.delta_L) < 1e-5 and np.linalg.norm(self.delta_Psi) < 1e-5:
          break
      # Update f
      self.update_f()
      if np.linalg.norm(self.delta_f) < 1e-5:
        break

        iteration += 1
        
    # Return L and f after optimization
    return self.L, self.f