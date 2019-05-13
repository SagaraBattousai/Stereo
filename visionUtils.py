import numpy as np
import cv2
import glob
import os 

getBaseDirectory = lambda : os.path.dirname(os.path.realpath(__file__))
    
def getPointsFromCamera(img):

  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  objp = np.zeros((6*7, 3), np.float32)
  objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)
    
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
  ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
    
  imgpoint = None

  if ret:
    #objpoints.append(objp)
    
    imgpoint = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    #imgpoints.append(corners2)

  return objp, imgpoint #, gray.shape

def getPointsFromImageDir(camera="", calibrationDir="cameraCalibrationImages", imageType="png"):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((6*7, 3), np.float32)
    objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)
    
    objpoints = []
    imgpoints = []
    
    images = glob.glob(getBaseDirectory() + "/" + calibrationDir + "/" + camera + "*." + imageType)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
    
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

    return objpoints, imgpoints, gray.shape



def calibrateCamera(camera="", calibrationDir="cameraCalibrationImages",
                    imageType="png", save=False, points=None):
   
    if points is None:
      objpoints, imgpoints, imgShape = getPointsFromImageDir(camera, calibrationDir, imageType)
    else:
      objpoints, imgpoints, imgShape = points

    calibration = cv2.calibrateCamera(objpoints, imgpoints, imgShape[::-1], None, None)
    
    if save:
        ret, mtx, dist, rvecs, tvecs = calibration
        
        np.savez(camera + "CameraCalibration.npz",
                 ret=np.array(ret, dtype=np.float32),
                 mtx=mtx,
                 dist=dist,
                 rvecs=np.array(rvecs),
                 tvecs=np.array(tvecs))

    return calibration

#temp (cyclic dep)
import stereoVision as sv

def calibrateStereoCamera(calibrationDir="cameraCalibrationImages",
                          imageType="png", save=False, leftCalibration=None, rightCalibration=None):
    
    objpoints, leftImgPoints, imgShape = getPointsFromImageDir("left", calibrationDir, imageType)
    _        , rightImgPoints, _       = getPointsFromImageDir("right", calibrationDir, imageType)

    left = leftCalibration if leftCalibration is not None else sv.CalibratedCamera(*calibrateCamera(
                               "left", calibrationDir=calibrationDir, imageType=imageType, save=save,
                                points=(objpoints, leftImgPoints, imgShape)))

    right = rightCalibration if rightCalibration is not None else sv.CalibratedCamera(*calibrateCamera(
                                "right", calibrationDir=calibrationDir, imageType=imageType, save=save,
                                 points=(objpoints, rightImgPoints, imgShape)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    (_, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
        objpoints, leftImgPoints, rightImgPoints,
        left.mtx, left.dist,
        right.mtx, right.dist,
        imgShape, None, None, None, None,
        cv2.CALIB_FIX_INTRINSIC, criteria)#TERMINATION_CRITERIA)

    OPTIMIZE_ALPHA = 1

    (leftRectification, rightRectification, leftProjection, rightProjection,
        disparityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
            left.mtx, left.dist,
            right.mtx, right.dist,
            imgShape, rotationMatrix, translationVector,
            None, None, None, None, None,
            cv2.CALIB_ZERO_DISPARITY, 0)#OPTIMIZE_ALPHA)

    leftMapX, leftMapY = cv2.initUndistortRectifyMap(left.mtx, left.dist, 
                                                     leftRectification, leftProjection,
                                                     imgShape, cv2.CV_32FC1)

    rightMapX, rightMapY = cv2.initUndistortRectifyMap(right.mtx, right.dist,
                                                       rightRectification, rightProjection,
                                                       imgShape, cv2.CV_32FC1)


    #Can't tell what I need Tutorial aint amazing so ill save their stuff for now and we'll see!
    if save:
    #    ret, mtx, dist, rvecs, tvecs = calibration
      np.savez("StereoCameraCalibration.npz",
               imgShape=imgShape,
               leftMapX=leftMapX,
               leftMapY=leftMapY,
               leftROI=leftROI,
               rightMapX=rightMapX,
               rightMapY=rightMapY,
               rightROI=rightROI)

    return (leftMapX, leftMapY, leftROI), (rightMapX, rightMapY, rightROI), imgShape


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 5)
    return img


def pose(camera="", calibrationDir="cameraCalibrationImages", imageType="png"):

    with np.load(camera + "CameraCalibration.npz") as cameraCalibration:
        mtx = cameraCalibration['mtx']
        dist = cameraCalibration['dist']

    objpoints, imgpoints, _ = getPointsFromImageDir(camera, calibrationDir, imageType)

    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

    images = glob.glob(getBaseDirectory() + "/" + calibrationDir + "/" + camera + "*." + imageType)

    for index, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners = imgpoints[index]

        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objpoints[index], corners, mtx, dist)

        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img, corners, imgpts)
        cv2.imshow('img', img)
        k = cv2.waitKey(0) & 0XFF

        # if k == 's':
        #     cv2.imwrite(fname[:6]+

    cv2.destroyAllWindows()


def undistort(img, mtx, dist):

  h, w = img.shape[:2]
  newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

  #undistort
  dst = cv2.undistort(img, mtx, dist, None, newCameraMtx)

  x,y,w,h = roi
  # print(x,y,w,h,dst.shape, end="\n\n",sep="\n")
  dst = dst[y:y+h, x:x+w]

  return dst

def fixedToFloatingPoint(arr, fixedPoint=4, dtype=np.float32):
  
  def perElem(elem):
    decimal = elem >> fixedPoint

    mask = ((1 << fixedPoint) - 1) & elem

    fraction = 0.0

    for i in range(fixedPoint, -1, -1):

      fraction += (mask & 0x01) * (1 / (1 << i))

      mask = mask >> 1

    return decimal + fraction

  return np.array(list(map(perElem, arr))).astype(dtype)


