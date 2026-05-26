import cv2
import numpy as np
def impulse_noise(image, noise_percent, salt_and_pepper):
    size = image.size   
    num_salt = np.ceil(noise_percent * size *   salt_and_pepper).astype('int')
    num_pepper = np.ceil(noise_percent * size * (1 - salt_and_pepper)).astype('int')
    row, column = img.shape
    x = np.random.randint(0, column - 1, num_pepper)
    y = np.random.randint(0, row - 1, num_pepper)
    image [y, x] = 0   #加入胡椒雜訊點(黑色)
    x = np.random.randint(0, column - 1, num_salt)
    y = np.random.randint(0, row - 1, num_salt)
    image [y, x] = 255 #加上鹽巴雜訊點(白色)
    return image

noise_percent = 0.05    # 雜訊在影像中的比例(影像中有10%的雜訊)
salt_and_pepper = 0.5  # 鹽與胡椒的比例(參數0.5為各50%)
img = cv2.imread('kpop.jpg', cv2.IMREAD_GRAYSCALE)
noisy = impulse_noise(img, noise_percent, salt_and_pepper)
cv2.imshow('Salt & Pepper Noise', noisy)
cv2.waitKey(0)
cv2.destroyAllWindows()

