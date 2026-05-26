import cv2
import numpy as np
from matplotlib import pyplot as plt

def gaussian_noise(image, mean, sigma):
    noise = np.random.normal(mean, sigma, image.shape)
    noise_img = np.clip(image + noise, 0, 255).astype('uint8')
    return noise_img, noise

mean = 0          
sigma = 30    
img = cv2.imread('kpop.jpg', cv2.IMREAD_GRAYSCALE)
noise_img, noise = gaussian_noise(img, mean, sigma)

plt.hist(noise.ravel(), 100, [-100, 100])
plt.show()

cv2.imshow('Original Image',  img)
cv2.imshow('Gaussian Noise',  noise_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

