import numpy
import scipy
import scipy.optimize
import matplotlib
import matplotlib.pyplot
import cv2
import open3d



def extrinsicCalibration_target_singleView():




    '''
    Return:
        objpoints: . Shape (nPoses, nViews, nPoints, 3). nPoses indicates the number of Point Cloud-Camera correspondances used.
                    nViws indicates the number of camera views used. nPints are the number of 3D Points detected in the Checkerboard
                    3 indicates that each point has 3 Coordinates.
                    For example, the shape (3, 1, 35, 3) implies that there are 3 objects,
                    each with 1 view, containing 35 points, and each point has 3 coordinates.
    '''    



    # Define the dimensions of checkerboard
    CHECKERBOARD = (8-1, 6-1)

    # Create object points for the checkerboard pattern
    obj_points = numpy.zeros((1, CHECKERBOARD[0]* CHECKERBOARD[1], 3), numpy.float32)
    obj_points[0, :, :2] = numpy.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)


    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.


    nPoses = 3

    images = []
    pointClouds = []




    # Load the lidar and camera data
    for i in range(0, nPoses):
        
        #pointClouds.append(numpy.loadtxt("lidar_points.txt"))
        pointClouds.append(open3d.io.read_point_cloud("Calibration\\Data Calibration\\Calibration Example Data\\pointClouds\\full\\pose{}_full.pcd".format(i+1)))
        images.append(cv2.imread("Calibration\\Data Calibration\\Calibration Example Data\\images\\full\\pose{}.png".format(i+1)))


    # Load camera intrisic parameters
    camera_matrix = numpy.load("matrix_intrinsic_example_full.npz")['matrix']
    dist_coeffs = numpy.load("matrix_distortion_example_full.npz")['matrix']

    matrix_list = []

    for i in range(0, nPoses):

        
        # Find the checkerboard corners in the camera image
        ret, corners = cv2.findChessboardCorners(images[i], CHECKERBOARD)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(obj_points)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(images[i], CHECKERBOARD, corners, ret)

        

        # Find the lidar-camera transformation matrix
        ret, rvec, tvec = cv2.solvePnP(numpy.array(obj_points), numpy.array(corners), camera_matrix, dist_coeffs)

        # Create the transformation matrix from the rotation and translation vectors
        rmat, _ = cv2.Rodrigues(rvec)
        transform = numpy.hstack((rmat, tvec))
        #print("{}\n====================".format(i+1))
        #print(transform)
        matrix_list.append(transform)
    
    return objpoints, imgpoints, matrix_list









points_3D, points_2D, transformationMatrix_list = extrinsicCalibration_target_singleView()


# Average Otimization
transformationMatrix_average = numpy.mean(numpy.array(transformationMatrix_list), axis = 0)

print(transformationMatrix_average)



# Bundle Adjustment Otimization

# Load camera intrisic parameters
camera_matrix = numpy.load("matrix_intrinsic_example_full.npz")['matrix']
dist_coeffs = numpy.load("matrix_distortion_example_full.npz")['matrix']





def bundle_adjustment(intrinsic, dist, 3Dpoints, 2Dpoints, extrinsic):
    """
    Perform bundle adjustment on a set of 3D points and corresponding 2D image points
    using a known intrinsic matrix, distortion coefficients, and initial extrinsic estimate.
    
    Parameters
    ----------
    intrinsic: numpy.ndarray
        The intrinsic camera matrix, shape (3, 3)
    dist: numpy.ndarray
        The distortion coefficients, shape (5,) or (8,)
    3Dpoints: numpy.ndarray
        The 3D points in world coordinates, shape (nPoses, nViews, nPoints, 3)
    2Dpoints: numpy.ndarray
        The corresponding 2D image points, shape (nPoses, nViews, nPoints, 2)
    extrinsic: numpy.ndarray
        The initial extrinsic matrix estimate, shape (nPoses, nViews, 4, 4)

    Returns
    -------
    optimized_extrinsic: numpy.ndarray
        The optimized extrinsic matrix, shape (nPoses, nViews, 4, 4)
    """
    # Reshape inputs for optimization
    nPoses, nViews, nPoints, _ = 3Dpoints.shape
    3Dpoints = 3Dpoints.reshape((nPoses * nViews * nPoints, 3))
    2Dpoints = 2Dpoints.reshape((nPoses * nViews * nPoints, 2))
    extrinsic = extrinsic.reshape((nPoses * nViews, 4, 4))
    
    # Define optimization objective
    def objective(params, intrinsic, dist, 3Dpoints, 2Dpoints):
        extrinsic = params.reshape((nPoses * nViews, 4, 4))
        X = np.hstack([3Dpoints, np.ones((nPoses * nViews * nPoints, 1))])
        X = np.einsum("ij,pkj->pki", X, extrinsic)
        X = X[:, :2] / X[:, 2:]
        X = np.einsum("ij,ij->i", intrinsic, X.T).T
        X = X + np.dot(dist, np.hstack([X**2, X**3, np.ones((nPoses * nViews * nPoints, 1))]))
        X = X.ravel() - 2Dpoints.ravel()
        return X
    
    # Perform optimization
    params0 = extrinsic.reshape((nPoses * nViews * 16,))
    optimized_params = least_squares(objective, params0, jac="3-point", bounds=(0, 1),
                                     args=(intrinsic, dist, 3Dpoints, 2Dpoints))
    optimized_extrinsic = optimized_params.x.reshape((nPoses * nViews, 4, 4))
    
    # Reshape outputs
    optimized_extrinsic = scipy.optimize.least_squares(bundle_adjustment_residual, initial_extrinsic, args=(K, distortion, 3d_points, 2d_points), method='lm')
    optimized_extrinsic = optimized_extrinsic.x.reshape(num_views, 4, 4)

    def bundle_adjustment_residual(extrinsic, K, distortion, 3d_points, 2d_points):
        num_poses, num_views, num_points, _ = 3d_points.shape
        extrinsic = extrinsic.reshape(num_views, 4, 4)
        residuals = []

        for pose in range(num_poses):
            for view in range(num_views):
                for point in range(num_points):
                    X = 3d_points[pose, view, point, :]
                    x = 2d_points[pose, view, point, :]
                    X_homo = numpy.hstack([X, numpy.ones((X.shape[0], 1))])
                    x_proj = numpy.dot(K, numpy.dot(extrinsic[view], X_homo.T))
                    x_proj = x_proj / x_proj[2]
                    residual = x_proj[:2] - x.T
                    residuals.append(residual)

    return numpy.array(residuals).ravel()








'''


def bundle_adjustment_singleView(X, x, K, T_init, k):
    """
    Perform bundle adjustment on the extrinsic calibration matrix.
    
    X : (N, 3) array of 3D points
    x : (N, 2) array of corresponding 2D points
    K : (3, 3) array of intrinsic camera matrix
    T_init : (4, 4) array of initial extrinsic calibration matrix
    k : (3,) array of radial distortion coefficients
    """
    a,b,c,d = X.shape
    # X.shape >> (3, 1, 35, 3)

    X = numpy.transpose(X, (2, 1, 0, 3))
    X = numpy.squeeze(X, axis=1)
    X = numpy.reshape(X, (c, -1, 3))
    # X.shape >> (35, -1, 3)
    
    a,b,c,d = x.shape
    # x.shape >> (3, 35, 1, 2)
    x = numpy.transpose(x, (1, 0, 3, 2))
    x = numpy.reshape(x, (b, -1, 2))
    # x.shape >> (35, -1, 2)

    def project(T, X):
        """
        Project 3D points onto the image plane.
        """
        X = numpy.hstack([X, numpy.ones((X.shape[0], 1))])
        X = numpy.dot(T, X.T).T[:, :3]
        x = X[:, 0] / X[:, 2]
        y = X[:, 1] / X[:, 2]
        r2 = x**2 + y**2
        radial = (1 + k[0] * r2 + k[1] * r2**2 + k[2] * r2**3)
        x_proj = K[0, 0] * x * radial + K[0, 2]
        y_proj = K[1, 1] * y * radial + K[1, 2]
        return numpy.stack([x_proj, y_proj], axis=1)
    
    def residuals(T, X, x):
        """
        Compute the reprojection error between the projected points and the observed 2D points.
        """
        x_proj = project(T, X)
        return (x - x_proj).ravel()
    
    T, _ = scipy.optimize.least_squares(residuals, T_init.ravel(), args=(X, x), method='lm')
    T = T.reshape(4, 4)
    return T



print(numpy.array(points_3D).shape)
print("=======================================")
print(numpy.array(points_2D).shape)

transformationMatrix_final = bundle_adjustment_singleView(numpy.array(points_3D), numpy.array(points_2D),
                                                        numpy.array(camera_matrix), numpy.array(transformationMatrix_initial),
                                                        numpy.array(dist_coeffs))

print(transformationMatrix_final)



###########################################################


# Bundle Adjustment

# Define the initial guess for the transformation matrix
transform = numpy.identity(4)

# Define the optimization function
def optimize_transform(x):
    
    rmat = cv2.Rodrigues(x[0:3])[0]
    tvec = x[3:6]
    transform = numpy.hstack((rmat, tvec))
    error = 0
    for i in range(0, nPoses):
        lidar_points = numpy.array(pointClouds[i].points)
        lidar_points = numpy.hstack((lidar_points, numpy.ones((lidar_points.shape[0], 1))))
        camera_points = cv2.projectPoints(lidar_points, rmat, tvec, camera_matrix, dist_coeffs)
        error += numpy.sum((camera_points - imgpoints[i]) ** 2)
    return error






from scipy.optimize import least_squares

# Define the objective function for bundle adjustment
def objective_function(params, objpoints, imgpoints, camera_matrix, dist_coeffs):
    
    rvec, tvec = params[:3], params[3:]
    rmat, _ = cv2.Rodrigues(rvec)
    transform = numpy.hstack((rmat, tvec))

    # Transform lidar points to camera coordinates
    projected_points, _ = cv2.projectPoints(objpoints, rvec, tvec, camera_matrix, dist_coeffs)

    # Compute the residuals
    residuals = imgpoints - projected_points.reshape(-1, 2)
    return residuals.ravel()



# Initialize the parameters for bundle adjustment
params_initial = numpy.zeros(6)
params_initial[:3] = cv2.Rodrigues(transform[:3, :3])[0].ravel()
params_initial[3:] = transform[:, 3]

# Perform bundle adjustment
params_optimized = least_squares(objective_function, params_initial, args=(objpoints, imgpoints, camera_matrix, dist_coeffs))

# Get the optimized parameters
rvec, tvec = params_optimized.x[:3], params_optimized.x[3:]
rmat, _ = cv2.Rodrigues(rvec)
transform_optimized = numpy.hstack((rmat, tvec))

# Print the optimized transformation matrix
print(transform_optimized)




# Save the lidar-camera transformation matrix to a file
#numpy.savez("camera_params.npz", transform=transform)




'''