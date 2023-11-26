import cv2
import numpy as np

def compute_gradient(source, type='x', ddepth = cv2.CV_64F):
    if(type == 'x'):
        return cv2.Sobel(source,ddepth,dx=1,dy=0)
    if(type == 'xx'):
        return cv2.Sobel(source,ddepth,dx=2,dy=0)
    if(type == 'y'):
        return cv2.Sobel(source,ddepth,dx=0,dy=1)
    if(type == 'yy'):
        return cv2.Sobel(source,ddepth,dx=0,dy=2)
    if(type == 'xy'):
        return cv2.Sobel(source,ddepth,dx=1,dy=1)
      
# https://github.com/aboucaud/pypher/blob/master/pypher/pypher.py#L283
def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.

    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.

    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.

    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array

    Returns
    -------
    otf : `numpy.ndarray`
        OTF array

    Notes
    -----
    Adapted from MATLAB psf2otf function

    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf

def Phi_func(x, threshold, k = 2.7, a = 6.1e-4, b = 5):
    return np.where(x <= threshold, -k*x, -b-a*x**2)