import sys
import cv2
print("Hello World\n")
print("I am python version: " + sys.version)

img = cv2.VideoCapture(0, cv2.CAP_DSHOW)
img.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
img.set(cv2.CAP_PROP_FRAME_WIDTH, 600)

while(True):
    ret, image = img.read()
    cv2.imshow('webCam', image)
    if(cv2.waitKey(1) & 0xFF == 27): break

img.release()
cv2.destroyAllWindows()