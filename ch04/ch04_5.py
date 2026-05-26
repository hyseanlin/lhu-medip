import cv2 

def sobel(image, dx_weight=0.5): 
	kernel_size=(3,3) 
	#高斯模糊化 
	blur_img = cv2.GaussianBlur(image, kernel_size, 0 ) 
	#水平方向梯度 
	x = cv2.Sobel(blur_img, cv2.CV_16S, 1, 0, kernel_size) 
	abs_x = 	cv2.convertScaleAbs(x) 
	#垂直方向梯度 
	y = cv2.Sobel(blur_img, cv2.CV_16S, 0, 1, kernel_size) 
	abs_y = cv2.convertScaleAbs(y) 
	#合併兩個方向的梯度 
	sobel_image = cv2.addWeighted( abs_x, dx_weight,abs_y, 1.0-dx_weight, 0 ) 
	return sobel_image 

#主程式 
gray_img = cv2.imread("kpop.jpg", cv2.IMREAD_GRAYSCALE) 
sobel_image=sobel(gray_img) 
cv2.imshow("sobel",sobel_image) 

sobel_dx=sobel(gray_img, 1.0) 
cv2.imshow("sobel with dx",sobel_dx) 

sobel_dy=sobel(gray_img, 0.0) 
cv2.imshow("sobel with dy",sobel_dy) 

cv2.waitKey(0) 
cv2.destroyAllWindows() 
