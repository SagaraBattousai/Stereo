import numpy as np
import cv2
import glob

def startCapturing(camera):
    if not camera.isOpened():
        camera.open()

# def calibrateFromImage(img):
#     # img = cv2.imread(fname)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     ret, corners = cv2.findChessboardCorners(gray, (6,5), None)

#     if ret == True:
#         #objpoints.append(objp)

#         corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
#         #imgpoints.append(corners2)

#         img = cv2.drawChessboardCorners(img, (6,5), corners2, ret)
#         cv2.imshow('img', img)
       
#         if cv2.waitKey(1) & 0xFF == ord(' '):
#             continue

        




criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((5*6, 3),  np.float32)
objp[:,:2] = np.mgrid[0:6, 0:5].T.reshape(-1,2)

objpoints = []
imgpoints = []

# leftImages = []
# rightImages = []

#images = glob.glob('cameraCalibrationImages/left*.png')

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

    images = np.hstack((lGray, rGray))

    cv2.imshow('images', images)

    lRet, lCorners = cv2.findChessboardCorners(lGray, (6,5), None)
    rRet, rCorners = cv2.findChessboardCorners(rGray, (6,5), None)
        
    key = cv2.waitKey(25)

    if (lRet and rRet) == True:

        lCorners2 = cv2.cornerSubPix(lGray, lCorners, (11,11), (-1,-1), criteria)
        rCorners2 = cv2.cornerSubPix(rGray, rCorners, (11,11), (-1,-1), criteria)

        lFrame = cv2.drawChessboardCorners(lFrame, (6,5), lCorners2, lRet)
        rFrame = cv2.drawChessboardCorners(rFrame, (6,5), rCorners2, rRet)
        
        frames = np.hstack((lFrame, rFrame))

        cv2.imshow('frames', frames)
       
        # objpoints.append(objp)
        # imgpoints.append(corners2)

        print("captured!")

        cv2.imwrite('left_capture' + str(captureCount) + ".png", lGray)
        cv2.imwrite('right_capture' + str(captureCount) + ".png", rGray)
        captureCount += 1
        print(captureCount, end="\r")
            

    
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




cv2.destroyAllWindows()
