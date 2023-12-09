import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_data(blur_path, sharp_path):
    blur = cv2.imread(blur_path, cv2.IMREAD_GRAYSCALE) # uint8
    blur_img = cv2.resize(blur, (300, 300)) # uint8
    
    sharp = cv2.imread(sharp_path, cv2.IMREAD_GRAYSCALE)  # uint8
    sharp_img = cv2.resize(sharp, (300, 300))  # uint8
    
    return blur_img, sharp_img

class Kernel:
    def __init__(self, blur_img, sharp_img, ker_size):
        self.blur = blur_img
        self.sharp = sharp_img
        self.k = ker_size
    
    def Calculate(self):
        F_blur = np.fft.fft2(blur_img)
        F_sharp = np.fft.fft2(sharp_img)

        F_kernel = F_blur / F_sharp
        kernel = np.fft.ifft2(F_kernel, (self.k, self.k))
        return kernel
    
    def reverse(self, kernel, sharp):
        # p = int((sharp.shape[0]-kernel.shape[0])/2)
        # kernel = np.pad(kernel, ((p,p),(p,p)), 'constant', constant_values = 0)
        F_kernel = np.fft.fft2(kernel, (sharp.shape[0], sharp.shape[1]))
        
        F_sharp = np.fft.fft2(sharp)
        F_blur = F_sharp * F_kernel
        blur = np.fft.ifft2(F_blur)
        return blur


blur_img, sharp_img = load_data("./data/test_dataset/blur.png", "./data/test_dataset/sharp.png")

plt.subplot(1,2,1)
plt.imshow(sharp_img, cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(blur_img, cmap = 'gray')
plt.savefig(f"./data/test_dataset/data.png")

plt.show()

a = Kernel(blur_img, sharp_img, 9)
kernel = a.Calculate()
plt.imshow(np.abs(kernel), cmap = 'gray')
plt.savefig(f"./data/test_dataset/kernel.png")
plt.show()

blur = a.reverse(kernel, sharp_img)
plt.subplot(1,2,1)
plt.imshow(blur_img, cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(np.abs(blur), cmap = 'gray')
plt.savefig(f"./data/test_dataset/reverse.png")
plt.show()
