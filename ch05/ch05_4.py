import cv2
import numpy as np

def salt_pepper_noise(image, noise_percent, salt_and_pepper):
    size = image.size
    num_salt = np.ceil(noise_percent * size * salt_and_pepper).astype('int')
    num_pepper = np.ceil(noise_percent * size * (1 - salt_and_pepper)).astype('int')
    row, column = image.shape
    x = np.random.randint(0, column - 1, num_pepper)
    y = np.random.randint(0, row - 1, num_pepper)
    image[y, x] = 0
    x = np.random.randint(0, column - 1, num_salt)
    y = np.random.randint(0, row - 1, num_salt)
    image[y, x] = 255
    return image

noise_percent = 0.1
salt_and_pepper = 0.5
img = cv2.imread('kpop.jpg', cv2.IMREAD_GRAYSCALE)

noise_img = salt_pepper_noise(img.copy(), noise_percent, salt_and_pepper)

cv2.imshow('Original', img)
cv2.imshow('Salt & Pepper Noise', noise_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

