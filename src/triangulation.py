"""
Implements various methods to triangulate a 3D point given 2D observations from multiple cameras.
"""

import numpy as np
from scipy.optimize import least_squares

MEASUREMENT_SIZE = 2


def linear_triangulate(camera_poses_world, normalized_features, min_dist=0.1, max_condition_num=10000):
    """Triangulate a point with the linear triangulation method(DLT).

    Args:
        camera_poses_world: List of the SE(3) camera poses.
        normalized_features: Observations of 3D point in normalized keypoint coordinates(u,v,1.0)
        min_dist: The minimum distance a point must be from a camera. Used as a validity check on the triangulated
            point. From most normal use cases(pinhole camera) we know that it is impossible for a 3D point to be behind
            a camera (Z is negative). By increasing the 'min_dist' > 0 we are making the check even stricter.
        max_condition_num: The maximum allowed condition number of our linear system. The condition number describes
            how well our linear system constrains the problem. (E.g camera with 0 displacement between them can't
            triangulate a point. and thus will have a high condition number). The equation to compute it is
            max_singular_value/smallest_singular_value.

    Returns:
        boolean value: A boolean value. True if triangulation was succesfull, and False on Failure
        triangulated_point: The triangulated point in 3D.

    This class implements a linear triangulation or Direct Linear Transform(DLT) to triangulate a 3D point. It does
    this by approximating the triangulation problem with a Linear System, which can then be solved with a basic least
    squares algorithms.

    For an overview of this method, and how to build the equations I recommend.
    http://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf

    Our method differs from the above derivation as follows:
    1. Our points are already in the normalized camera coordinates so the camera matrix P is just the extrinsics. The
        intrinsics are not needed.
    2. We setup the linear system in the non homogenous form(Ax=b). The homogenous form moves the b vector into the A
        matrix so that the equation looks something like this [A|b]x=0 and is used in the above reference.

    We also implement some basic validity checking such as ensuring that the point is some distance in front of all the
    cameras, and that our linear system is well conditioned.
    """
    assert (len(camera_poses_world) >= 2)
    assert (len(camera_poses_world) == len(normalized_features))

    num_measurements = len(camera_poses_world)

    # Our linear system matrix Ax=b
    A = np.empty((2 * num_measurements, 3), dtype=np.float64)
    b = np.empty((2 * num_measurements), dtype=np.float64)

    for idx, (pose, feature) in enumerate(zip(camera_poses_world, normalized_features)):
        cRw = pose.R.T
        wPc = pose.t
        cPw = -cRw @ wPc
        u = feature[0]
        v = feature[1]

        A[MEASUREMENT_SIZE * idx + 0] = u * cRw[2] - cRw[0]
        A[MEASUREMENT_SIZE * idx + 1] = v * cRw[2] - cRw[1]

        b[MEASUREMENT_SIZE * idx + 0] = -(u * cPw[2] - cPw[0])
        b[MEASUREMENT_SIZE * idx + 1] = -(v * cPw[2] - cPw[1])

    # Solve our linear system(Ax=b) for x which is our triangulated point.
    result = np.linalg.lstsq(A, b, rcond=None)
    triangulated_point = result[0]

    # Check the condition number. The singular values are sorted in descending order.
    singular_values = result[3]
    condition_number = singular_values[0] / singular_values[2]
    if condition_number > max_condition_num:
        return False, triangulated_point

    # Check that the triangulated point is in front of all the cameras by some distance.
    for pose in camera_poses_world:
        cRw = pose.R.T
        wPc = pose.t
        camera_t_world = -cRw @ wPc
        transformed_pt = cRw @ triangulated_point + camera_t_world
        if (transformed_pt[2] <= min_dist):
            return False, triangulated_point

    return True, triangulated_point


def convert_point3d_to_inverse_depth(pt3d):
    """ Convert a point into the inverse depth format.

    Args:
        pt3d: A point in 3D coordinates(X,Y,Z)

    Returns:
        Point in inverse depth format (alpha,beta,rho)

    This function converts a point into the inverse depth format. As discussed in A.I. Mourikis, S.I. Roumeliotis:
     "A Multi-state Constraint Kalman Filter for Vision-Aided Inertial Navigation".

    This format ends up being more numerically stable, which is usefull when triangulating it.
    """
    z = pt3d[2]
    return np.array([pt3d[0] / z, pt3d[1] / z, 1.0 / z])


def convert_inverse_depth_to_point3d(inverse_depth_pt):
    """Converts an inverse depth point back into a standard 3D point(X,Y,Z)

    Args:
        inverse_depth_pt: 3x1 numpy array(alpha,beta,rho)

    Returns:
        A point in 3D

    """
    rho = inverse_depth_pt[2]
    return np.array([inverse_depth_pt[0] / rho, inverse_depth_pt[1] / rho, 1.0 / rho])


def compute_error_and_jacobian(inverse_depth_pt, camera_pose_world, measurement, compute_jacobian=False):
    """Given a point, a camera pose, and a feature measurement compute the error and jacobian.

    Args:
        inverse_depth_pt: Landmark in inverse depth format(alpha,beta,rho).
        camera_pose_world: Camera pose in SE(3).
        measurement: Measurement of the landmark in normalized camera coordinates.
        compute_jacobian: Flag whether to compute the jacobian.

    Returns:

    The equations for this function can be found in T. Hinzmann, "Robust Vision-Based Navigation for Micro Air
     Vehicles" and also contains the jacobians.

    """
    alpha = inverse_depth_pt[0]
    beta = inverse_depth_pt[1]
    rho = inverse_depth_pt[2]

    cRw = camera_pose_world.R.T
    wPc = camera_pose_world.t
    cPw = -cRw @ wPc

    h = cRw @ np.array([alpha, beta, 1.0]) + (rho * cPw)

    projected_pt = np.array([h[0] / h[2], h[1] / h[2]])
    reprojection_error = projected_pt - measurement[0:2]

    W = np.zeros((3, 3))
    W[:, 0:2] = cRw[:, 0:2]
    W[:, 2] = cPw

    if (compute_jacobian):
        jac = np.empty((2, 3))
        jac[0] = 1.0 / h[2] * W[0] - h[0] / (h[2]**2) * W[2]
        jac[1] = 1.0 / h[2] * W[1] - h[1] / (h[2]**2) * W[2]
        return reprojection_error, jac
    else:
        return reprojection_error

def model(estimated_inverse_depth_pt, camera_poses_world, measurements):
    """Model function for scipy optimizer which triangulates a 3D point.

    Args:
        estimated_inverse_depth_pt: Current estimate of landmark in the inverse depth format.
        camera_poses_world: List of camera poses that observed this Landmark in SE(3).
        measurements: List of normalized keypoint measurement. Same length as 'camera_poses_world'

    Returns:
        The residuals or error of the current estimate(landmark) given the observations(measurements)

    This function is used in conjunction with scipy's least squares optimizer. It computes the error of the given
    estimate.
    """
    assert (len(camera_poses_world) == len(measurements))
    residuals = np.empty(len(measurements) * 2)
    for idx, (pose, measurement) in enumerate(zip(camera_poses_world, measurements)):
        error = compute_error_and_jacobian(estimated_inverse_depth_pt, pose, measurement, False)
        residuals[idx * 2 + 0] = error[0]
        residuals[idx * 2 + 1] = error[1]
    return residuals


def compute_jacobian_scipy(estimated_inverse_depth_pt, camera_poses_world, measurements):
    """Function to compute jacobians of triangulating a 3D point. Meant to be used with scipy optimizer.

    Args:
        estimated_inverse_depth_pt: Current estimate of landmark in the inverse depth format.
        camera_poses_world: List of camera poses that observed this Landmark.
        measurements: List of normalized keypoint measurement. Same length as 'camera_poses_world'

    Returns:
        Numpy array representing the jacobians. Should be of size len(measurements)*2 x 3.

    """
    jac = np.empty((len(measurements) * 2, 3))
    for idx, (pose, measurement) in enumerate(zip(camera_poses_world, measurements)):
        _, jac_i = compute_error_and_jacobian(estimated_inverse_depth_pt, pose, measurement, True)
        index = idx * 2
        jac[index:index + 2] = jac_i
    return jac


def optimize_point_location(initial_pt3d, camera_poses_world, normalized_features):
    """Optimizes the point location using a non linear least squares optimizer.

    Args:
        initial_pt3d: Initial estimate of the point location in 3D.
        camera_poses_world: List of camera poses that observed this Landmark 
        normalized_features: List of normalized keypoint measurement. Same length as 'camera_poses_world'

    Returns:
        boolean value: True on success, else False
        Optimized point location.

    This function uses scipy's least squares solver, which itself implements uses a Levenberg Marquadt implementation
    from MINPACK.
    """
    inverse_depth_pt = convert_point3d_to_inverse_depth(initial_pt3d)

    result = least_squares(model,
                           inverse_depth_pt,
                           jac=compute_jacobian_scipy,
                           args=(camera_poses_world, normalized_features),
                           verbose=0)

    return result.success, convert_inverse_depth_to_point3d(result.x)

