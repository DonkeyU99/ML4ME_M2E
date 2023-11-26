#def std_convoluted(image, N):
#im = np.array(image, dtype=float)
#     im2 = im**2
    #     ones = np.ones(im.shape)
    
    #     kernel = np.ones((2*N+1, 2*N+1))
    #     s = scipy.signal.convolve2d(im, kernel, mode="same")
    #     s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
    #     ns = scipy.signal.convolve2d(ones, kernel, mode="same")
    
#     return np.sqrt((s2 - s**2 / ns) / ns)

def smooth_region(img, kernel_size, threshold):
    local_std = np.zeros(img.shape)
    std_kernel = np.ones((kernel_size, kernel_size))
    im = np.array(img, dtype = float)
    im2 = im**2
    ones = np.ones((im.shape[1], im.shape[2]))

    for i in range(3):
        s = scipy.signal.convolve2d(im[i], std_kernel, mode="same")
        s2 = scipy.signal.convolve2d(im2[i], std_kernel, mode="same")
        ns = scipy.signal.convolve2d(ones, std_kernel, mode="same")
        local_std[i] = np.sqrt((s2 - s**2 / ns) / ns)
        
    region = (local_std[0] > threshold[0]) & (local_std[1] > threshold[1]) & (local_std[2] > threshold[2])

    return region # True면 Omega에 들어가는 pixel (img.shape[1], img.shape[2])