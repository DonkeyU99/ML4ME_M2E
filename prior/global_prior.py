import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import scipy
from utils import compute_gradient
import os

def global_prior(image, weights, mu, cov):
    gradx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    grady = cv2.Sobel(image, cv2.CV_64F, 0, 1)

    # Calculate the magnitude of the gradient
    grad_magnitude = np.sqrt(gradx**2 + grady**2)
    grad_sign = gradx * grady
    grad_magnitude[grad_sign < 0] = -grad_magnitude[grad_sign < 0]
    
    log_prior = 0
    for grad in grad_magnitude:
        prior = 0
        for i in range(len(mu)):
            prior += weights[i]*scipy.stats.logpdf((grad-mu[i])/np.sqrt(cov))
        log_prior+=np.log(prior)
        
class prior_g_GM():
    def __init__(self, n_components,random_state=42):
        self.model = GaussianMixture(n_components=n_components, random_state=random_state)
        self.gaussian_weights = None

    def fit(self,img_path='data/toy_dataset',plot=False):
        dir_list = os.listdir(img_path)

        hist_values_all = []

        for i in dir_list:
            img=cv2.imread(os.path.join(img_path,i))
            img2= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gradx = compute_gradient(img2,'x')
            grady = compute_gradient(img2,'y')

            # Calculate the magnitude of the gradient
            grad_magnitude = np.sqrt(gradx**2 + grady**2)
            # Convert to integers
            grad_sign = gradx * grady
            grad_magnitude[grad_sign < 0] = -grad_magnitude[grad_sign < 0]

            #as int
            grad_magnitude_int = grad_magnitude.astype(int)

            # Flatten the gradients
            flattened_grad = grad_magnitude_int.flatten()
        
            hist_values, bin_edges = np.histogram(flattened_grad, bins=range(-1000, 1001))
            hist_values_all.append(hist_values)
        
        hist_values_all = np.array(hist_values_all)

        # Flatten the gradients if you want a 1D array
        if(plot):
            hist_plots = hist_values_all.sum(axis=0)
            normalize = hist_values_all.sum()
            plt.bar(bin_edges[:-1], np.log(hist_plots/normalize), width=1, color='blue', alpha=0.7)
            plt.title('Density Distribution of Flattened Gradient Magnitude')
            plt.xlabel('Gradient Magnitude (int)')
            plt.ylabel('Density')
            plt.show()

        self.model.fit(hist_values_all)
        self.gaussian_weights = self.model.weights_
        
    def predict(self,img_path='data/toy_dataset/14_IPHONE-7_M.jpeg',plot=False):
        img=cv2.imread(img_path)
        img2= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gradx = compute_gradient(img2,'x')
        grady = compute_gradient(img2,'y')

        # Calculate the magnitude of the gradient
        grad_magnitude = np.sqrt(gradx**2 + grady**2)
        grad_sign = gradx * grady
        grad_magnitude[grad_sign < 0] = -grad_magnitude[grad_sign < 0]

        # Convert to integers
        grad_magnitude_int = grad_magnitude.astype(int)
    
        # Flatten the gradients
        flattened_grad = grad_magnitude_int.flatten()
        hist_values, bin_edges = np.histogram(flattened_grad, bins=range(-1000, 1001))
        hist_values = hist_values.reshape(-1, 1)

        if(plot):
            # Flatten the gradients if you want a 1D array
            num_pixels = img2.shape[0]*img2.shape[1]
            plt.bar(bin_edges[:-1], np.log(hist_values/num_pixels), width=1, color='blue', alpha=0.7)
            plt.title('Density Distribution of Flattened Gradient Magnitude')
            plt.xlabel('Gradient Magnitude (int)')
            plt.ylabel('Density')
            plt.show()

        predicted_proba = global_prior(img2, self.gaussian_weights, self.model.means_, self.model.covariances_ )

        return predicted_proba