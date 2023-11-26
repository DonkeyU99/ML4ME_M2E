import numpy
import numpy.fft.fft as fft
from utils import compute_gradient

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
  
  def likelihood_update():
    gradients = ['x','y','xx','xy','yy']
    gradient_filters = compute_gradient(gradients)
    Delta = np.sum(omega(len(gradients))*np.dot(conj_fft(gradient_filters),gradient_filters))
    
    grad_x = compute_gradient(noise,src=-1,dx=1,dy=0)
    grad_y = compute_gradient(noise,src=-1,dx=0,dy=1)

    numer = np.sum(np.dot(conj_fft(kernel),fft(image),Delta)+gamma*np.dot(conj_fft(grad_x),fft(phi_x))+gamma*np.dot(conj_fft(grad_y),fft(phi_y)))
    denom = np.sum(np.dot(conj_fft(kernel),fft(kernel),Delta)+gamma*np.dot(conj_fft(grad_x),fft(grad_x))+gamma*np.dot(conj_fft(grad_y),fft(grad_y)))
    self.new_L = np.fft.ifft(numer/denom)
    
  def filter_update():  
    
  return self.new_L