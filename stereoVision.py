import numpy as np
import cv2
import glob

def calibrateCamera(camera="", baseDir="cameraCalibrationImages",
                    imageType="png", save=False):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((5*6, 3), np.float32)
    objp[:,:2] = np.mgrid[0:6, 0:5].T.reshape(-1,2)
    
    objpoints = []
    imgpoints = []
    
    images = glob.glob(baseDir + "/" + camera + "*." + imageType)
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        ret, corners = cv2.findChessboardCorners(gray, (6,5), None)
    
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
    
            # img = cv2.drawChessboardCorners(img, (6,5), corners2, ret)
            # cv2.imshow('img', img)
            # while True:
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
    #cv2.destroyAllWindows()


    calibration = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    if save:
        ret, mtx, dist, rvecs, tvecs = calibration
        
        np.savez(camera + "CameraCalibration.npz",
                 ret=np.array(ret, dtype=np.float32),
                 mtx=mtx,
                 dist=dist,
                 rvecs=np.array(rvecs),
                 tvecs=np.array(tvecs))

    return calibration


    
    
    
