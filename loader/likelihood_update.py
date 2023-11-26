import numpy as np

class likelihood_update(likelihood, zeta_0, gamma,tau, sigma):
  def __init__(self):
    self.new_L = None
    self.lambda_1 = 1/tau;
    self.lambda_2 = 1/(sigma**2*tau);
    self.omega = None;

  def calculate_omega():
    q = [0,1,2]
    self.omega = 1/(zeta_0**2*tau*2**q)
    
  def update():
    delta = np.sum()
    self.new_L = np.fft.ifft()
    
  return self.new_L