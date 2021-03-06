import numpy as np

def project_point_pose(pose, point_in_world):

    cRw = pose.R.T
    wPc = pose.t

    pt_in_camera = cRw @ (point_in_world - wPc)

    return pt_in_camera / pt_in_camera[2]

def invert_pose(pose):
    new_rot = pose[0:3, 0:3].T
    trans = pose[0:3, 3]
    new_pose = np.eye(4, dtype=np.float64)
    new_pose[0:3, 0:3] = new_rot
    new_pose[0:3, 3] = -new_rot @ trans
    return new_pose


def compute_relative_rot12(world_SE3_apple, world_SE3_banana):
    rot_apple = world_SE3_apple[0:3, 0:3]
    rot_banana = world_SE3_banana[0:3, 0:3]

    return rot_apple.T @ rot_banana


def compute_relative_T12(world_SE3_apple, world_SE3_banana):
    return invert_pose(world_SE3_apple) @ world_SE3_banana


def compute_relative_trans12(world_SE3_apple, world_SE3_banana):
    t_apple = world_SE3_apple[0:3, 3]
    t_banana = world_SE3_banana[0:3, 3]
    world_R_apple = world_SE3_apple[0:3, 0:3]
    apple_R_world = world_R_apple.T
    return apple_R_world @ (-t_apple + t_banana)

def project_point(world_SE3_camera, pt_in_world):
    camera_SE3_world = invert_pose(world_SE3_camera)
    pt_in_camera = camera_SE3_world[0:3, 0:3] @ pt_in_world + camera_SE3_world[0:3, 3]
    pt_in_camera /= pt_in_camera[2]
    return pt_in_camera


def compute_3x4_mat(rotation, translation):
    t = np.empty((3, 4))
    t[0:3, 0:3] = rotation
    t[0:3, 3] = translation
    return t

def to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    Pay attention to the convention used. The function follows the
    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
    A Tutorial for Quaternion Algebra", Equation (78).
    The input quaternion should be in the form [q1, q2, q3, q4(scalar)]
    """
    if R[2, 2] < 0:
        if R[0, 0] > R[1, 1]:
            t = 1 + R[0,0] - R[1,1] - R[2,2]
            q = [t, R[0, 1]+R[1, 0], R[2, 0]+R[0, 2], R[1, 2]-R[2, 1]]
        else:
            t = 1 - R[0,0] + R[1,1] - R[2,2]
            q = [R[0, 1]+R[1, 0], t, R[2, 1]+R[1, 2], R[2, 0]-R[0, 2]]
    else:
        if R[0, 0] < -R[1, 1]:
            t = 1 - R[0,0] - R[1,1] + R[2,2]
            q = [R[0, 2]+R[2, 0], R[2, 1]+R[1, 2], t, R[0, 1]-R[1, 0]]
        else:
            t = 1 + R[0,0] + R[1,1] + R[2,2]
            q = [R[1, 2]-R[2, 1], R[2, 0]-R[0, 2], R[0, 1]-R[1, 0], t]

    q = np.array(q) # * 0.5 / np.sqrt(t)
    return q / np.linalg.norm(q)