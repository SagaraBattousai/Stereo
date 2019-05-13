import numpy as np
import cv2
import glob
from os.path import isfile as fileExists

import visionUtils as utils

class CalibratedCamera():

  def __init__(self, ret, mtx, dist, rvecs, tvecs):
    self.ret = ret
    self.mtx = mtx
    self.dist = dist
    self.rvecs = rvecs
    self.tvecs = tvecs

  def undistort(self, img):
    mtx = self.mtx
    dist = self.dist

    h, w = img.shape[:2]
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    #undistort
    dst = cv2.undistort(img, mtx, dist, None, newCameraMtx)

    # x,y,w,h = roi
    # dst = dst[y:y+h, x:x+w]

    return dst
  

class StereoCamera():

  def __init__(self, left=None, right=None):
    if left is not None:
      self.left = left
    elif fileExists("leftCameraCalibration.npz"):
      self.leftCalibrationFilename = "leftCameraCalibration.npz"
      with np.load(self.leftCalibrationFilename) as leftCalibration:
        self.left = CalibratedCamera(**leftCalibration)
    else:
      leftCalibration = utils.calibrateCamera("left", save=True)
      self.left = CalibratedCamera(*leftCalibration)

    if right is not None:
      self.right = right
    elif fileExists("rightCameraCalibration.npz"):
      self.rightCalibrationFilename = "rightCameraCalibration.npz"
      with np.load(self.rightCalibrationFilename) as rightCalibration:
        self.right = CalibratedCamera(**rightCalibration)
    else:
      rightCalibration = utils.calibrateCamera("right", save=True)
      self.right = CalibratedCamera(*rightCalibration)




def reprojectionError(cameraName=""):
    
  with np.load(cameraName + "CameraCalibration.npz") as cameraCalibration:
    camera = CalibratedCamera(**cameraCalibration)
  
  mean_error = 0
  
  objpoints, imgpoints, _ = utils.getPointsFromImageDir(cameraName)

  for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], camera.rvecs[i], camera.tvecs[i], camera.mtx, camera.dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
  

  print("Total error: {}".format(mean_error/len(objpoints)))


if __name__ == "__main__":

  reprojectionError("left")
  reprojectionError("right")

  
  # vision = StereoCamera()

  # mtx = vision.left.mtx
  # dist = vision.left.dist

  # img = cv2.imread(utils.getBaseDirectory() + "/" + 'right_capture27.png')
  # h, w = img.shape[:2]
  # newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

  # #undistort
  # dst = cv2.undistort(img, mtx, dist, None, newCameraMtx)

  # x,y,w,h = roi
  # dst = dst[y:y+h, x:x+w]
  
  # cv2.imshow("img", img)
  # cv2.imshow("dst", dst)

  # cv2.waitKey(0)
    








