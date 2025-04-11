from ObjectDetection import *

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
#cap.set(10,70)

while True:
    success, img = cap.read()
    results,objectInfo = getObjects(img,object==['person'])
    #print(objectInfo)
    cv2.imshow("lena", img)
    cv2.waitKey(0)