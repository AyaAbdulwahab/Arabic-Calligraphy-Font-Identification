import cv2
import numpy as np
from scipy.signal import convolve2d
from skimage.filters import gaussian

class Preprocessing:

    @staticmethod
    def gray_scale_img(img):
        # Convert the image to gray scale
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Then apply an otsu threshold (automatic thresholding) to the image
        th, im_gray_th_otsu = cv2.threshold(im_gray, 0, 255,  cv2.THRESH_OTSU)

        return im_gray_th_otsu


    @staticmethod
    # La place of gaussian
    def log(image):
        # Apply gaussian filters with sigma 0.3
        gaussian_img = gaussian(image, sigma=0.3)*255

        # Chosen laplacian filter
        log_1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

        # Convolute with the image
        img_1 = convolve2d(gaussian_img, log_1)

        
        thres= 100
        img_1[np.abs(img_1) < thres] = 1
        img_1[np.abs(img_1) >= thres] = 0

        return img_1