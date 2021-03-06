import numpy as np
from transforms3d.euler import euler2mat

from src.spatial_transformations import Isometry3d
from src.triangulation import linear_triangulate, optimize_point_location
from tests.utilities import project_point_pose

def test_linear_triangulate():
    point3d = np.array([0.15, 0.2, 3.1])
    rotation_matrices = [euler2mat(0.1, .2, .4), euler2mat(0.3, -.1, .25), euler2mat(0.05, 0.5, 0.3)]
    translations = [np.array([0.1, 0.2, -.4]), np.array([-0.25, .3, 0.35]), np.array([0.6, .1, -.1])]

    poses = []
    normalized_points = []

    for rot_mat, trans in zip(rotation_matrices, translations):
        pose = Isometry3d(rot_mat, trans)
        normalized_measurement = project_point_pose(pose, point3d)
        poses.append(pose)
        normalized_points.append(normalized_measurement)

    _, triangulated_point = linear_triangulate(poses, normalized_points)
    assert (np.allclose(triangulated_point, point3d))
    is_valid, triangulated_point = optimize_point_location(point3d, poses, normalized_points)
    assert (is_valid)
    assert (np.allclose(triangulated_point, point3d))

if __name__ == "__main__": 
    test_linear_triangulate()