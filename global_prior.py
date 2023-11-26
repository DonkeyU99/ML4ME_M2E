import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from utils import compute_gradient

class prior_g_GM():
    def __init__(self, n_components,random_state=42):
        self.model = GaussianMixture(n_components=n_components, random_state=random_state)
        self.gaussian_weights = None

    def fit(self,img_path='data/toy_dataset/0_IPHONE-SE_M.JPG',plot=False):
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

        # Flatten the gradients if you want a 1D array
        hist_values, bin_edges = np.histogram(flattened_grad, bins=range(min(flattened_grad), max(flattened_grad) + 1))

        if(plot):
            num_pixels = img2.shape[0]*img2.shape[1]
            plt.bar(bin_edges[:-1], np.log(hist_values/num_pixels), width=1, color='blue', alpha=0.7)
            plt.title('Density Distribution of Flattened Gradient Magnitude')
            plt.xlabel('Gradient Magnitude (int)')
            plt.ylabel('Density')
            plt.show()
        
        hist_values = hist_values.reshape(-1, 1)
        self.model.fit(hist_values)
        self.gaussian_weights = self.model.weights_

    def predict(self,img_path='data/toy_dataset/14_IPHONE-7_M.jpeg'):
        img=cv2.imread(img_path)
        img2= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gradx = compute_gradient(img2,'x')
        grady = compute_gradient(img2,'y')

        # Calculate the magnitude of the gradient
        grad_magnitude = np.sqrt(gradx**2 + grady**2)
        grad_sign = gradx * grady
        grad_magnitude[grad_sign < 0] = -grad_magnitude[grad_sign < 0]
    
        # Flatten the gradients
        flattened_grad = grad_magnitude_int.flatten()

        # Flatten the gradients if you want a 1D array
        hist_values, bin_edges = np.histogram(flattened_grad, bins=range(min(flattened_grad), max(flattened_grad) + 1))
        hist_values = hist_values.reshape(-1, 1)

        predicted_proba = self.model.predict_proba(hist_values)

        return predicted_proba