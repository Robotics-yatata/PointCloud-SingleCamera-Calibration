import numpy
import cv2
import matplotlib
import matplotlib.pyplot
import glob
import os


def camera_calibration(img_dir:str = None, alpha:int = 0, return_images:bool = False, return_roi:bool = False)->tuple:
    """
    Perform camera intrinsic calibration using Corner detector.
    
    Parameters:
    img_dir (str) : path to the image directory
    return_images (bool) : flag to return list of images with red circles on the detected corner positions
    alpha (float) : the free scaling parameter between 0 (when all pixels in the undistorted image will be valid)
                     and 1 (when all pixels in the undistorted image will be zero)
    
    Returns:
    tuple : containing the optimal new camera matrix, distortion coefficients, rotation vectors, translation vectors,
            list of images with red circles on the detected corner positions if return_images is True
            and the Region of Interest if return_roi is True { x, y, w, h = roi; dst = dst[y:y+h, x:x+w] }
    
    Dependencies :
    - opencv-python
    - numpy
    - os
    """

    # Define the dimensions of checkerboard
    CHECKERBOARD = (8-1, 6-1)


    # Vector for 3D points
    objpoints = []
 
    # Vector for 2D points
    imgpoints = []


    #  3D points real world cooROInates
    objectp3d = numpy.zeros((1, CHECKERBOARD[0]* CHECKERBOARD[1], 3), numpy.float32)
    objectp3d[0, :, :2] = numpy.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)


    # Load the images from the given directory
    images = []
    for file in os.listdir(img_dir):
        if file.endswith(".png") or file.endswith(".jpg"):
            img_path = os.path.join(img_dir, file)
            if not os.path.isfile(img_path):
                raise ValueError(f"{img_path} not found.")
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"{img_path} failed to load.")
        images.append(img)
    
    # Create an empty list to store the corners
    corners = []
    images_with_corners = []

    # Iterate through each image and detect the corners
    for img in images:
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
                        
        # Refining pixel coordinates
        # for given 2d points.  
        if ret == True:
        
            objpoints.append(objectp3d)
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))   
            imgpoints.append(corners)
        
        if return_images:
            # Draw red circles on the detected corner positions
            img_with_corners = img.copy()
            for i in zip(*corners[::-1]):
                cv2.circle(img_with_corners,i,5,(0,0,255),-1)
            images_with_corners.append(img_with_corners)


    # Perform camera calibration using the detected corners
    if len(imgpoints) == 0:
        print("No checkerboard patterns were found in the images.")
    else:
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
     
    
    if not ret:
        raise Exception("Calibration failed.")

    # Get the optimal new camera matrix
    # h,  w = img.shape[:2]
    # newcameramtx, roi = cv.getOptimalNewCameraMatrix(matrix, distortion, (w,h), 1, (w,h)) 
    matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, gray.shape[::-1], 0)

    if return_images and return_roi:
        return matrix, distortion, r_vecs, t_vecs, images_with_corners, roi
    elif return_images:
        return matrix, distortion, r_vecs, t_vecs, images_with_corners
    elif return_roi:
        return matrix, distortion, r_vecs, t_vecs, roi
    else:
        return matrix, distortion, r_vecs, t_vecs




path_full = r"Calibration\Data Calibration\Calibration Example Data\images\full"
path_target = r"Calibration\Data Calibration\Calibration Example Data\images\target"

matrix, distortion, r_vecs, t_vecs = camera_calibration(img_dir=path_full, alpha = 0, return_images = False, return_roi = False)


# Displaying required output
print(" Camera matrix:")
print(matrix)
 
print("\n Distortion coefficient:")
print(distortion)
 
print("\n Rotation Vectors:")
print(r_vecs)
 
print("\n Translation Vectors:")
print(t_vecs)



# Save the matrix to a ".npz" file
numpy.savez("matrix_intrinsic_example_full.npz", matrix=matrix)
numpy.savez("matrix_distortion_example_full.npz", matrix=distortion)

