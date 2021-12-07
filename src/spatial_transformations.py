from math import sqrt
import numpy as np

def hamiltonian_quaternion_to_rot_matrix(q, eps=np.finfo(np.float64).eps):
    w, x, y, z = q
    squared_norm = np.dot(q, q)
    if squared_norm < eps:
        return np.eye(3)
    s = 2.0 / squared_norm
    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z
    return np.array([[1.0 - (yy + zz), xy - wz, xz + wy], [xy + wz, 1.0 - (xx + zz), yz - wx],
                     [xz - wy, yz + wx, 1.0 - (xx + yy)]])

class SO3Pose():
    """Represents a rigid body transform using SO(3) for rotation.

    A rigid body transform is composed of a rotation + translation. In this class we store the rotation as a
     SO(3). 
    """
    def __init__(self, R, trans):
        self.R = R
        self.t = trans

    def rotation(self):
        return self.R

    def translation(self):
        return self.t

    @classmethod
    def from_rotation_matrix_and_trans(cls, rotation_matrix, translation):
        return SO3Pose(rotation_matrix, translation)

    def transform_vector(self, vec):
        """Transform a point by the rigid body transform.

        Args:
            vec: 3x1 or 4x1 numpy array. If 4x1 then the point needs to be in homogenous form([x,y,z,1.0])

        Returns:
            Numpy array representing the transformed point.
        """
        if vec.size != 3 or vec.size != 4:
            raise TypeError("Point size must be 3 or 4 when transforming it")
        rot_mat = self.R
        return rot_mat @ vec[0:3] + self.t

    def multiply_transform(self, other):
        new_R = self.R @ other.R 
        new_t = self.R @ other.t + self.t
        return SO3Pose(new_R, new_t)

    def __matmul__(self, other):
        pass

    def __repr__(self):
        return "SO3Pose: " +\
            "\n| rot:" + str(self.R) + \
            "\n| trans:" + str(self.t)
