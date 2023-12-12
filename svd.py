import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_data_gray(blur_path, sharp_path):
    blur = cv2.imread(blur_path, cv2.IMREAD_GRAYSCALE) # uint8
    blur_img = cv2.resize(blur, (blur.shape[1],blur.shape[0])) # uint8
    
    sharp = cv2.imread(sharp_path, cv2.IMREAD_GRAYSCALE)  # uint8
    sharp_img = cv2.resize(sharp, (sharp.shape[1],blur.shape[0]))  # uint8
    
    return blur_img, sharp_img

def load_data(blur_path, sharp_path):
    blur = cv2.imread(blur_path) # uint8
    blur_img = cv2.resize(blur, (blur.shape[1],blur.shape[0])) # uint8
    
    sharp = cv2.imread(sharp_path)  # uint8
    sharp_img = cv2.resize(sharp, (sharp.shape[1],blur.shape[0]))  # uint8
    
    return blur_img, sharp_img

class Kernel:
    def __init__(self, blur_img, sharp_img, ker_size):
        self.blur = blur_img
        self.sharp = sharp_img
        self.k = ker_size
        self.s_size =(1000,1000)
 
    def Calculate(self):
        pad_size = int(self.k/2)
        F_blur = np.fft.fft2(np.pad(self.blur,((pad_size,pad_size),(pad_size,pad_size))),(2000,2000))
        F_sharp = np.fft.fft2(np.pad(self.sharp,((pad_size,pad_size),(pad_size,pad_size))),(2000,2000))

        F_kernel = F_blur / F_sharp
        kernel = np.fft.ifft2(F_kernel, (2000,2000))
        kernel = kernel[:self.k,:self.k]
        return kernel
    
    def Calculate_Reverse(self):
        pad_size = int(self.k/2)
        F_blur = np.fft.fft2(np.pad(self.blur,((pad_size,pad_size),(pad_size,pad_size))),(2000,2000))
        F_sharp = np.fft.fft2(np.pad(self.sharp,((pad_size,pad_size),(pad_size,pad_size))),(2000,2000))

        F_kernel = F_sharp / F_blur
        kernel = np.fft.ifft2(F_kernel, (2000,2000))
        kernel = kernel[:self.k,:self.k]
        return kernel

    def sharp_to_blur(self, kernel, sharp):
        # p = int((sharp.shape[0]-kernel.shape[0])/2)
        # kernel = np.pad(kernel, ((p,p),(p,p)), 'constant', constant_values = 0)
        F_kernel = np.fft.fft2(kernel,(2000,2000))
        
        pad_size = int(self.k/2)
        img = np.zeros_like(sharp)
        for i in range(3):
            F_sharp = np.fft.fft2(np.pad(sharp[:,:,i],((pad_size,pad_size),(pad_size,pad_size))),(2000,2000))
            F_blur = F_sharp * F_kernel
            blur = np.fft.ifft2(F_blur,(2000,2000))*np.sqrt(2*np.pi)
            img[:,:,i] = blur[:sharp.shape[0],:sharp.shape[1]]
        return img

    def blur_to_sharp(self, inv_kernel, blur):
        # p = int((sharp.shape[0]-kernel.shape[0])/2)
        # kernel = np.pad(kernel, ((p,p),(p,p)), 'constant', constant_values = 0)
        F_kernel = np.fft.fft2(inv_kernel,(2000,2000))
        
        pad_size = int(self.k/2)
        img = np.zeros_like(blur)
        for i in range(3):
            F_blur = np.fft.fft2(np.pad(blur[:,:,i],((pad_size,pad_size),(pad_size,pad_size))),(2000,2000))
            F_sharp = F_blur * F_kernel
            sharp = np.fft.ifft2(F_sharp,(2000,2000))*np.sqrt(2*np.pi)
            img[:,:,i] = sharp[:blur.shape[0],:blur.shape[1]]
        return img

blur_img_gray, sharp_img_gray = load_data_gray("./data/test_dataset/blur1.png","./data/test_dataset/sharp1.png")

plt.subplot(1,2,1)
plt.imshow(sharp_img_gray, cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(blur_img_gray, cmap = 'gray')
plt.savefig(f"./data/test_dataset/data.png")

plt.show()

a = Kernel(blur_img_gray, sharp_img_gray, 35)
kernel = a.Calculate()
reverse_kernel = a.Calculate_Reverse()
plt.imshow(np.abs(kernel), cmap = 'gray')
plt.savefig(f"./data/test_dataset/kernel.png")
plt.imshow(np.abs(reverse_kernel), cmap = 'gray')
plt.savefig(f"./data/test_dataset/reverse_kernel.png")
plt.show()


blur_img, sharp_img = load_data("./data/test_dataset/blur1.png","./data/test_dataset/sharp1.png")

blur = a.sharp_to_blur(kernel, sharp_img)
plt.subplot(1,2,1)
plt.imshow(blur_img)
plt.subplot(1,2,2)
plt.imshow(np.abs(blur))
plt.savefig(f"./data/test_dataset/sharp_to_blur.png")
plt.show()

blur = a.blur_to_sharp(reverse_kernel,blur_img)
plt.subplot(1,2,1)
plt.imshow(blur_img)
plt.subplot(1,2,2)
plt.imshow(np.abs(blur))
plt.savefig(f"./data/test_dataset/blur_to_sharp.png")
plt.show()