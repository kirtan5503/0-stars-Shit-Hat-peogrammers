import cv2
 
img_read = cv2.imread('itslinuxfoss.png')
img_blur = cv2.blur(img_read, (5,5))
cv2.imshow('Image',img_blur)