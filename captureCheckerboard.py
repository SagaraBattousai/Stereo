import numpy as np
import cv2

def startCapturing(camera):
  if not camera.isOpened():
    camera.open()

def captureCheckerboard(leftIndex=2, rightIndex=3):

  leftCamera = cv2.VideoCapture(leftIndex)
  rightCamera = cv2.VideoCapture(rightIndex)
  
  startCapturing(leftCamera)
  startCapturing(rightCamera)
  
  captureCount = 1
  
  while True:
    lRet, lFrame = leftCamera.read()
    rRet, rFrame = rightCamera.read()
  
    # lGray = cv2.cvtColor(lFrame, cv2.COLOR_BGR2GRAY)
    # rGray = cv2.cvtColor(rFrame, cv2.COLOR_BGR2GRAY)
  
    grayz = np.hstack((lFrame, rFrame))
  
    cv2.imshow('Framez', grayz)
  
    key = cv2.waitKey(1)
  
    if key & 0xFF == ord(' '):
      cv2.imwrite('left_capture' + str(captureCount) + ".png", lFrame)
      cv2.imwrite('right_capture' + str(captureCount) + ".png", rFrame)
      captureCount += 1
    
    if key & 0xFF == ord('q'):
      break
  
  cap.release()
  cv2.destroyAllWindows()

def captureCheckerboardAnd(func, leftIndex=2, rightIndex=3):

  leftCamera = cv2.VideoCapture(leftIndex)
  rightCamera = cv2.VideoCapture(rightIndex)
  
  startCapturing(leftCamera)
  startCapturing(rightCamera)
  
  captureCount = 1
  
  while True:
    lRet, lFrame = leftCamera.read()
    rRet, rFrame = rightCamera.read()
  
    captureCount = func(lFrame, rFrame, captureCount)
  
    key = cv2.waitKey(1)
  
    # if key & 0xFF == ord(' '):
    #   cv2.imwrite('left_capture' + str(captureCount) + ".png", lFrame)
    #   cv2.imwrite('right_capture' + str(captureCount) + ".png", rFrame)
    #   captureCount += 1
    
    if key & 0xFF == ord('q'):
      break
  
  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  captureCheckerboard()



