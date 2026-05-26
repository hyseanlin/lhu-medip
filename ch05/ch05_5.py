import cv2
import numpy as np
import matplotlib.pyplot as plt

def white_noise(image, floor_noise, ceiling_noise):
    noise = np.random.randint(floor_noise, ceiling_noise, image.shape)
    noisy_img=np.clip(image + noise , 0 ,255).astype('uint8')
    noise = noise.astype('uint8')
    return noise, noisy_img

img = cv2.imread('kpop.jpg', cv2.IMREAD_GRAYSCALE)


noise_low = 0
noise_high = 90
#設定雜訊像素值的限制，最低以及最高像素值
noise, noisy_img = white_noise(img, noise_low, noise_high)

plt.hist(noise.ravel(), noise_high-noise_low+1, [noise_low, noise_high])
plt.show()

cv2.imshow('Original Image', img)
cv2.imshow('Image w/ White Noise', noisy_img)
cv2.imshow('White Noise', noise)
cv2.waitKey(0)
cv2.destroyAllWindows()

