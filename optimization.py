import numpy
import numpy.fft.fft as fft
from utils import compute_gradient, psf2otf

class optimizer():
  def __init__(self, likelihood, zeta_0, gamma, tau, sigma, image, kernel):
    self.new_L = None
    self.lambda_1 = 1/tau;
    self.lambda_2 = 1/(sigma**2*tau);
    self.omega = None;

  def omega(q):
    return 1/((zeta_0**2)*tau*(2**q))
    
  def conj_fft(array): # conj(FFT) operator
    return np.conj(np.fft.fft(array))
  
  def gradient_filter(type):
    #TODO
    
  def likelihood_update():
    gradients = ['x','y','xx','xy','yy']
    gradient_filters = gradient_filter(gradients)
    phi_x = compute_gradient(likelihood, 'x')
    phi_y = compute_gradient(likelihood, 'y')
    grad_x = gradient_filter('x')
    grad_y = gradient_filter('y')
    
    Delta = np.sum(omega(len(gradients))*conj_fft(gradient_filters)*gradient_filters)     # q=1 for x y, q=2 for xx xy yy

    numer = np.sum( conj_fft(kernel)*fft(image)*Delta + gamma*conj_fft(grad_x)*fft(phi_x) + gamma*conj_fft(grad_y)*fft(phi_y) )
    numer = np.sum( conj_fft(kernel)*fft(kernel)*Delta + gamma*conj_fft(grad_x)*fft(grad_x) + gamma*conj_fft(grad_y)*fft(grad_y) )
    self.new_L = np.fft.ifft(numer/denom)
    return new_L
  
  def filter_update():  
    # TODO
    return new_filter