import numpy
import numpy.fft.fft as fft
from utils import compute_gradient

class likelihood_update():
  def __init__(self, likelihood, zeta_0, gamma,tau, sigma, image, kernel):
    self.new_L = None
    self.lambda_1 = 1/tau;
    self.lambda_2 = 1/(sigma**2*tau);
    self.omega = None;

  def calculate_omega():
    q = [0,1,2]
    self.omega = 1/(zeta_0**2*tau*2**q)
    
  def conj_fft(array): # conj(FFT) operator
    return np.conj(np.fft.fft(array))
  
  def update():
    delta = np.sum(dot_fft(gradients))
    
    grad_x = compute_gradient(noise,src=-1,dx=1,dy=0)
    grad_y = compute_gradient(noise,src=-1,dx=0,dy=1)

    numer = np.sum(np.dot(conj_fft(kernel),fft(image),delta)+gamma*np.dot(conj_fft(grad_x),fft(phi_x))+gamma*np.dot(conj_fft(grad_y),fft(phi_y)))
    denom = np.sum(np.dot(conj_fft(kernel),fft(kernel),delta)+gamma*np.dot(conj_fft(grad_x),fft(grad_x))+gamma*np.dot(conj_fft(grad_y),fft(grad_y)))
    self.new_L = np.fft.ifft(numer/denom)
    
  return self.new_L