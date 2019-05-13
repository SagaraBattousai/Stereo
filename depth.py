import numpy as np
import cv2
from matplotlib import pyplot as plt

import stereoVision as sv
import visionUtils as vu


def startCapturing(camera):
    if not camera.isOpened():
        camera.open()


def exampleDepth():

  # downscale images for faster processing
  imgL = cv2.pyrDown(cv2.imread(cv2.samples.findFile('aloeL.jpg')))
  imgR = cv2.pyrDown(cv2.imread(cv2.samples.findFile('aloeR.jpg')))


  cv2.imshow("imgl", imgL)

  # disparity range is tuned for 'aloe' image pair
  window_size = 3
  min_disp = 16
  num_disp = 112-min_disp
  stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
  numDisparities = num_disp,
  blockSize = 16,
  P1 = 8*3*window_size**2,
  P2 = 32*3*window_size**2,
  disp12MaxDiff = 1,
  uniquenessRatio = 10,
  speckleWindowSize = 100,
  speckleRange = 32
  )
  
  #print('computing disparity...')
  disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

  plt.imshow(disp,'gray')
  #plt.title("nd = " + str(nd) + ", bs = " + str(bs))
  plt.show()


def depthWithFilter(leftImg, rightImg):
    
  lGray = cv2.cvtColor(leftImg, cv2.COLOR_BGR2GRAY)
  rGray = cv2.cvtColor(rightImg, cv2.COLOR_BGR2GRAY)
        
  vision = sv.StereoCamera()
    
  left = vision.left
  right = vision.right

  # lUndistOrig = left.undistort(lGray)
  # rUndistOrig = right.undistort(rGray)

  lUndist = left.undistort(lGray)
  rUndist = right.undistort(rGray)

  # lUndist = cv2.pyrDown(lUndistOrig)
  # rUndist = cv2.pyrDown(rUndistOrig)

  # numDis = [16, 32, 64, 128]
  # blocksize = [5,7,11]

  # best = (32, 7)#?
  # best = (32, 5)#?
  # best = (64, 5)#?

  SBM = []

  settings = []

  # for nd in numDis:
  #   for bs in blocksize:

  for (nd, bs) in [(32,7),(32,5),(64,5)]:
      
    stereo = cv2.StereoBM_create(numDisparities=nd, blockSize=bs)
      
      #disparity = vu.fixedToFloatingPoint(stereo.compute(lUndist, rUndist))
    disparity = stereo.compute(lUndist, rUndist).astype(np.float32) / 16.0

    disparity[disparity < 0] = 0

    #blured = cv2.GaussianBlur(disparity, (5, 5), 0)
      
    #blured = cv2.medianBlur(disparity, 5)
    blured = cv2.bilateralFilter(disparity, 9, 75, 75)

    SBM.append((disparity, blured))
    settings.append((nd,bs))

  key = cv2.waitKey(1)

  if key & 0xFF == ord('q'):
      pass
  #break

  #cv2.destroyAllWindows()
  return SBM, settings







if __name__ == "__main__":


  # leftImg = cv2.imread('StereoBuildIkeaLeft.png')
  # rightImg = cv2.imread('StereoBuildIkeaRight.png', 0)
  
  leftImg = cv2.imread('StereoDepthLeft.png')
  rightImg = cv2.imread('StereoDepthRight.png')

  SBM, settings = depthWithFilter(leftImg, rightImg)


  for ind, (i, f) in enumerate(SBM):
    print("nd: " + str(settings[ind][0]) + " bs: " + str(settings[ind][1]))
    plt.subplot(121),plt.imshow(i, 'gray'),plt.title('Origional')
    plt.xticks([]),  plt.yticks([])
    plt.subplot(122),plt.imshow(f, 'gray'),plt.title('Blurred')
    plt.xticks([]),  plt.yticks([])
    plt.show()










  # imgl = cv2.imread('StereoBuildIkeaLeft.png')
  # imgr = cv2.imread('StereoBuildIkeaRight.png')
  
  # d1, d2 = staticDepth(imgl, imgr)

  # cv2.imshow('frame', np.hstack((imgl,imgr)))
 
  # for i, SBG in enumerate(d1):
  #   plt.imshow(SBG,'gray')
  #   plt.title("Origional")
  #   plt.show()
  #   plt.title("New")
  #   plt.imshow(d2[i],'gray')
  #   plt.show()


  #   if cv2.waitKey(1) == ord('q'):
  #     break

  
  # cap.release()
  # cv2.destroyAllWindows()
 
 # videoCapture() 
  
  # cap = cv2.VideoCapture('sceneVideos/pan.avi')

  # while cap.isOpened():

  #   ret, frame = cap.read()

  #   lframe = frame[:,:640,:]
  #   rframe = frame[:,640:,:]

  #   cv2.imshow('frame', frame)

  #   d1, d2 = staticDepth(lframe, rframe)

  #   for i, SBG in enumerate(d1):
  #     plt.imshow(SBG,'gray')
  #     plt.title("Origional")
  #     plt.show()
  #     plt.title("New")
  #     plt.imshow(d2[i],'gray')
  #     plt.show()


  #   if cv2.waitKey(1) == ord('q'):
  #     break

  
  # cap.release()
  # cv2.destroyAllWindows()




