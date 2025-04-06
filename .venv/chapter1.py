import cv2

img =cv2.imread("Resources/kitty.png")

imggGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imggBlur = cv2.GaussianBlur(imggGray,(5,5),0)

cv2.imshow("imgGray",imggGray)
cv2.imshow("imgBlur",imggBlur)
cv2.waitKey(0)