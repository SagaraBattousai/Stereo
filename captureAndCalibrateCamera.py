import cv2
import numpy as np
import glob
import os

import visionUtils as utils
import captureCheckerboard as cc

def checkerboard(leftImg, rightImg, captureCount):
  
  stereoImage = np.hstack((leftImg, rightImg))

  leftChess = leftImg.copy()
  rightChess = rightImg.copy()
  
  cv2.imshow('stereoImage', stereoImage)

  leftObjPoint, leftImgPoint = utils.getPointsFromCamera(leftImg)
  rightObjPoint, rightImgPoint = utils.getPointsFromCamera(rightImg)

  if leftImgPoint is not None and rightImgPoint is not None:
    
    cv2.drawChessboardCorners(leftChess, (7,6), leftImgPoint, True)
    cv2.drawChessboardCorners(rightChess, (7,6), rightImgPoint, True)

    cv2.imshow('leftChess', leftChess)
    cv2.imshow('rightChess', rightChess)

    cv2.imwrite("left_capture{}.png".format(captureCount), leftImg)
    cv2.imwrite("right_capture{}.png".format(captureCount), rightImg)
  
    captureCount += 1

  return captureCount

def checkerboardIndividual(leftImg, rightImg, captureCount):
  
  if captureCount == 1:
    leftcc = 1
    rightcc = 1
  else:
    rightcc = captureCount & 0xFFFF
    leftcc = captureCount >> 16

  stereoImage = np.hstack((leftImg, rightImg))

  leftChess = leftImg.copy()
  rightChess = rightImg.copy()
  
  cv2.imshow('stereoImage', stereoImage)

  leftObjPoint, leftImgPoint = utils.getPointsFromCamera(leftImg)
  rightObjPoint, rightImgPoint = utils.getPointsFromCamera(rightImg)

  if leftImgPoint is not None:
    cv2.drawChessboardCorners(leftChess, (7,6), leftImgPoint, True)
    cv2.imshow('leftChess', leftChess)

    cv2.imwrite('left_capture' + str(leftcc) + ".png", leftImg)
    leftcc += 1
  
  if rightImgPoint is not None:
    cv2.drawChessboardCorners(rightChess, (7,6), rightImgPoint, True)
    cv2.imshow('rightChess', rightChess)
  
    cv2.imwrite('right_capture' + str(rightcc) + ".png", rightImg)
    rightcc += 1

  return (leftcc << 16) | rightcc

def checkCalibrationImages(cameraSide="", calibrationDir="cameraCalibrationImages", imageType="png"):

  images = glob.glob(utils.getBaseDirectory() + "/" + calibrationDir + "/" + cameraSide + "*." + imageType)

  for fname in images:
    img = cv2.imread(fname)
    objpoint, imgpoint = utils.getPointsFromCamera(img)
    
    if imgpoint is not None:
      cv2.drawChessboardCorners(img, (7,6), imgpoint, True)
      cv2.imshow(fname, img)

      key = cv2.waitKey(0)

      if key & 0xFF == ord('d'):
        os.remove(fname)
        cv2.destroyWindow(fname)
        continue

      if key & 0xFF == ord('c'):
        cv2.destroyWindow(fname)
        continue

      if key & 0xFF == ord('q'):
        break

  cv2.destroyAllWindows()


def checkCalibrationPairs(imageCount=None, calibrationDir="cameraCalibrationImages", imageType="png"):
    
  base_calibration_dir = utils.getBaseDirectory() + "/" + calibrationDir + "/"

  if imageCount is None:
    imageCount = len(os.listdir(base_calibration_dir)) // 2

  images_format = base_calibration_dir + "{camera}_capture{index}." + imageType

  for index in range(1, imageCount + 1):
    
    left_fname = images_format.format(camera="left", index=index)
    right_fname = images_format.format(camera="right", index=index)

    if not (os.path.exists(left_fname) and os.path.exists(right_fname)):
      continue
    
    left_img = cv2.imread(left_fname)
    right_img = cv2.imread(right_fname)

    _, left_imgpoint = utils.getPointsFromCamera(left_img)
    _, right_imgpoint = utils.getPointsFromCamera(right_img)

    if left_imgpoint is not None and right_imgpoint is not None:
      cv2.drawChessboardCorners(left_img, (7,6), left_imgpoint, True)
      cv2.drawChessboardCorners(right_img, (7,6), right_imgpoint, True)
      

    img = np.hstack((left_img, right_img))

    window_name = "Capture" + str(index)

    cv2.imshow(window_name, img)

    key = cv2.waitKey(0)

    if key & 0xFF == ord('d'):
      os.remove(left_fname)
      os.remove(right_fname)
      cv2.destroyWindow(window_name)
      continue

    if key & 0xFF == ord('c'):
      cv2.destroyWindow(window_name)
      continue

    if key & 0xFF == ord('q'):
      break

  cv2.destroyAllWindows()


if __name__ == '__main__':

  #cc.captureCheckerboardAnd(checkerboard)
  
  checkCalibrationPairs(40)


