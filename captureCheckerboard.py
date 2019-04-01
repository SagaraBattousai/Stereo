import numpy as np
import cv2

def startCapturing(camera):
    if not camera.isOpened():
        camera.open()

leftCamera = cv2.VideoCapture(2)
rightCamera = cv2.VideoCapture(3)

startCapturing(leftCamera)
startCapturing(rightCamera)

captureCount = 1

while True:
    lRet, lFrame = leftCamera.read()
    rRet, rFrame = rightCamera.read()

    lGray = cv2.cvtColor(lFrame, cv2.COLOR_BGR2GRAY)
    rGray = cv2.cvtColor(rFrame, cv2.COLOR_BGR2GRAY)

    grayz = np.hstack((lGray, rGray))

    cv2.imshow('lFrame', lGray)
    cv2.imshow('rFrame', rGray)
    cv2.imshow('Framez', grayz)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        cv2.imwrite('left_capture' + str(captureCount) + ".png", lGray)
        cv2.imwrite('right_capture' + str(captureCount) + ".png", rGray)
        captureCount += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
