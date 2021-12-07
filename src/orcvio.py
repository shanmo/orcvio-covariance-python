import logging
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from math import sqrt
import sophus as sp

import numpy as np
import scipy
from scipy.stats import chi2

from src.spatial_transformations import Isometry3d
from src.math_utilities import skew, symmeterize_matrix, odotOperator, Hl_operator, Jl_operator, get_cam_wrt_imu_se3_jacobian
from src.orcvio_types import FeatureTrack
from src.triangulation import linear_triangulate, optimize_point_location

logger = logging.getLogger(__name__)


class StateInfo():
    ''' Stores size and indexes related to the MSCKF state vector'''

    # Represents the indexing of the individual components  of the state vector.
    # The slice(a,b) class just represents a python slice [a:b]
    # R, v, p, bg, ba 
    ATT_SLICE = slice(0, 3)
    VEL_SLICE = slice(3, 6)  # Velocity
    POS_SLICE = slice(6, 9)  # Position
    BG_SLICE = slice(9, 12)  # Bias gyro
    BA_SLICE = slice(12, 15)  # Bias accelerometer

    IMU_STATE_SIZE = 15  # Size of the imu state(the above 5 slices)

    CLONE_START_INDEX = 15  # Index at which the camera clones start
    CLONE_STATE_SIZE = 6  # Size of the camera clone state(se3)

    CLONE_SE3_SLICE = slice(0, 6)  # Where within the camera clone the se3 error state is


class CameraClone():
    def __init__(self, R, t, camera_id):
        self.camera_pose_global = Isometry3d(R, t)
        self.timestamp = 0
        self.camera_id = camera_id


class State():
    """
    Contains the state vector and the associated covariance for our Kalman Filter.

    Within the state vector we keep track of our IMU State(contains position,attitude,biases,...) and a limited
    amount of Camera State Stochastic Clones. These clones contain the position and attitude of the camera at some
    timestamp in the past. They are what allow us to define EKF Update functions linking the past poses to our current
    pose.
    """
    def __init__(self):

        # Attitude of the IMU. 
        # Stores the rotation of the IMU to the global frame as SO(3).
        self.imu_R_global = np.eye(3)

        # Position of the IMU in the global frame.
        self.global_t_imu = np.zeros((3, ), dtype=np.float64)

        # Velocity of the IMU
        self.velocity = np.zeros((3, ), dtype=np.float64)
        self.bias_gyro = np.zeros((3, ), dtype=np.float64)
        self.bias_acc = np.zeros((3, ), dtype=np.float64)
        self.clones = OrderedDict()

        # The covariance matrix of our state.
        self.covariance = np.eye(StateInfo.IMU_STATE_SIZE, dtype=np.float64)

    def set_velocity(self, vel):
        assert (vel.size == 3)
        self.velocity = vel

    def set_gyro_bias(self, bias_gyro):
        assert (bias_gyro.size == 3)
        self.bias_gyro = bias_gyro

    def set_acc_bias(self, bias_acc):
        assert (bias_acc.size == 3)
        self.bias_acc = bias_acc

    def add_clone(self, camera_clone):
        self.clones[camera_clone.camera_id] = camera_clone

    def num_clones(self):
        return len(self.clones)

    def calc_clone_index(self, index_within_clones):
        return StateInfo.CLONE_START_INDEX + index_within_clones * StateInfo.CLONE_STATE_SIZE

    def update_state(self, delta_x):
        """Update the state vector given a delta_x computed from a measurement update.

        Args:
            delta_x: Numpy vector. Has length equal to the error state vector.
        """
        assert (delta_x.shape[0] == self.get_state_size())

        # For everything except for the rotations we can use a simple vector update
        # x' = x+delta_x

        self.global_t_imu += delta_x[StateInfo.POS_SLICE]
        self.velocity += delta_x[StateInfo.VEL_SLICE]
        self.bias_gyro += delta_x[StateInfo.BG_SLICE]
        self.bias_acc += delta_x[StateInfo.BA_SLICE]

        # Attitude requires a special Quaternion update
        # Note because we are using the right jacobians the update needs to be applied from the right side.
        self.imu_R_global = self.imu_R_global @ sp.SO3.exp(delta_x[StateInfo.ATT_SLICE]).matrix()

        # Now do same thing for the rest of the clones

        for idx, clone in enumerate(self.clones.values()):

            delta_x_index = StateInfo.IMU_STATE_SIZE + idx * StateInfo.CLONE_STATE_SIZE
            se3_start_index = delta_x_index + StateInfo.CLONE_SE3_SLICE.start
            se3_end_index = delta_x_index + StateInfo.CLONE_SE3_SLICE.stop
            delta_x_slice = delta_x[se3_start_index:se3_end_index]
            delta_wTc = sp.SE3.exp(delta_x_slice).matrix()
            delta_wTc = np.squeeze(delta_wTc)
            clone.camera_pose_global.t += np.squeeze(clone.camera_pose_global.R @ delta_wTc[:3,3])
            clone.camera_pose_global.R = clone.camera_pose_global.R @ delta_wTc[:3,:3]

    def get_state_size(self):
        return StateInfo.IMU_STATE_SIZE + self.num_clones() * StateInfo.CLONE_STATE_SIZE

    def print_state(self):
        print("Position {}", self.global_t_imu)

    def get_pos_covariance(self): 
        """Obtain the covariance for position 
        """
        return np.copy(self.covariance[StateInfo.POS_SLICE, StateInfo.POS_SLICE])

class ORCVIO():
    """ Implements the ORCVIO.
    """
    def __init__(self, params, camera_calibration):
        self.state = State()
        self.params = params
        self.imu_buffer = []
        self.map_id_to_feature_tracks = {}
        self.camera_id = 0
        self.camera_calib = camera_calibration
        self.gravity = np.array([0, 0, -9.81])
        self.noise_matrix = np.eye(15)
        self.chi_square_val = {}
        for i in range(1, 100):
            self.chi_square_val[i] = chi2.ppf(0.05, i)

    def set_imu_noise(self, sigma_gyro, sigma_acc, sigma_gyro_bias, sigma_acc_bias):
        """Set the noise/random walk parameters of the IMU sensor.

        Args:
            sigma_gyro: Standard deviation of the gyroscope measurement.
            sigma_acc: Standard deviation of the accelorometer measurement
            sigma_gyro_bias: Standard deviation of the gyroscope bias evolution
            sigma_acc_bias: Standard deviation of the accelerometer bias

        These parameters can typically be found in the datasheet associated with your IMU.
        https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model this link gives a good overview of what the parameters
        mean and how they can be found.
        """
        # Since it is the standard deviation we need to square the variable for the variance.
        self.noise_matrix[:3, :3] = np.eye(3) * sigma_gyro**2
        self.noise_matrix[3:6, 3:6] = np.eye(3) * sigma_acc**2
        self.noise_matrix[9:12, 9:12] = np.eye(3) * sigma_gyro_bias**2
        self.noise_matrix[12:15, 12:15] = np.eye(3) * sigma_acc_bias**2

    def initialize(self, global_R_imu, global_t_imu, vel, bias_acc, bias_gyro):
        self.state.imu_R_global = global_R_imu
        self.state.global_t_imu = global_t_imu
        self.state.velocity = vel
        self.state.bias_acc = bias_acc
        self.state.bias_gyro = bias_gyro

    def set_imu_covariance(self, att_var, pos_var, vel_var, bias_gyro_var, bias_acc_var):
        '''Used to set the covariance of the IMU section in the covariance matrix.

        Args:
            att_var:
            pos_var:
            vel_var:
            bias_gyro_var:
            bias_acc_var:

        This should typically be one of the first functions you call to initialize the system. It sets how confident we
        are in the initial values of the IMU(smaller is more confident). Note that even if you know a value perfectly,
        you should never set its covariance to 0 as it causes numerical problems. Instead set it to an extremely small
        value such as 1e-12.

        '''
        cov = self.state.covariance
        st = StateInfo
        arr = np.empty((st.IMU_STATE_SIZE))
        arr[st.ATT_SLICE] = att_var
        arr[st.POS_SLICE] = pos_var
        arr[st.VEL_SLICE] = vel_var
        arr[st.BG_SLICE] = bias_gyro_var
        arr[st.BA_SLICE] = bias_acc_var
        np.fill_diagonal(cov, arr)

    def remove_old_clones(self):
        """Remove old camera clones from the state vector.

        In order to keep our computation bounded we remove certain camera clones from our state vector. In this
        implementation we implement a basic sliding window which always removes the oldest camera clone. This way our
        state vector is kept to a constant size.
        """

        num_clones = self.state.num_clones()

        # If we have yet to reach the maximum number of camera clones then skip
        if num_clones < self.params.max_clone_num:
            return

        # Remove the oldest
        ids_to_remove = []
        oldest_camera_id = list(self.state.clones.keys())[0]

        # Run the update on any features which have this clone
        for id, track in self.map_id_to_feature_tracks.items():
            track_cam_id = track.camera_ids[0]
            if track_cam_id == oldest_camera_id:
                ids_to_remove.append(id)

        self.update(ids_to_remove)

        # Remove the clone from the state vector

        # Since it is the oldest it is the first clone within our covariance matrix.
        clone_start_index = StateInfo.CLONE_START_INDEX
        clone_end_index = clone_start_index + StateInfo.CLONE_STATE_SIZE
        s = slice(clone_start_index, clone_end_index)

        new_cov = np.delete(self.state.covariance, s, axis=0)
        new_cov = np.delete(new_cov, s, axis=1)
        assert (new_cov.shape[0] == new_cov.shape[1])
        self.state.covariance = new_cov

        del self.state.clones[oldest_camera_id]

    def add_camera_features(self, feature_ids, normalized_keypoints):
        """
        Args:
            feature_ids:
            normalized_keypoints:

        Returns:
        """
        mature_feature_ids = []
        newest_clone_id = self.augment_camera_state(0)
        for id, keypoint in zip(feature_ids, normalized_keypoints):
            if id not in self.map_id_to_feature_tracks:
                track = FeatureTrack(id, keypoint, self.camera_id)
                self.map_id_to_feature_tracks[id] = track
                continue

            # Id is in feature track
            track = self.map_id_to_feature_tracks[id]
            track.tracked_keypoints.append(keypoint)
            track.camera_ids.append(newest_clone_id)
            if len(track.tracked_keypoints) >= self.params.max_track_length:
                mature_feature_ids.append(id)

        lost_feature_ids = []
        for id, track in self.map_id_to_feature_tracks.items():
            if track.camera_ids[-1] != newest_clone_id:
                lost_feature_ids.append(id)

        ids_to_update = mature_feature_ids + lost_feature_ids
        self.update(ids_to_update)

    def check_feature_motion(self, id, min_motion_dist=0.05, use_orthogonal_dist=False):
        """Check if the camera poses observing the feature has sufficient displacement between them.

        Args:
            id:
            min_motion_dist:
            use_orthogonal_dist:

        Returns: True if there is enough movement.

        When triangulating a point we require that there is sufficient movement of the camera, to be able to triangulate
        it accurately. E.g if the camera just stood in place, there would be no baseline and it would be impossible
        to estimate the depth correctly.

        """
        track = self.map_id_to_feature_tracks[id]
        if len(track.tracked_keypoints) < 2:
            return False

        if not use_orthogonal_dist:
            first_camera_pose = self.state.clones[track.camera_ids[0]]
            second_camera_pose = self.state.clones[track.camera_ids[-1]]

            global_t_camera1 = first_camera_pose.camera_pose_global.t
            global_t_camera2 = second_camera_pose.camera_pose_global.t

            if np.linalg.norm(global_t_camera1 - global_t_camera2) > min_motion_dist:
                return True
            return False
        else:
            first_camera_pose = self.state.clones[track.camera_ids[0]]

            global_R_camera1 = first_camera_pose.camera_pose_global.R
            feature_vec = np.append(track.tracked_keypoints[0], 1)
            bearing_vec_camera = feature_vec / np.linalg.norm(feature_vec)
            bearing_in_global = global_R_camera1 * bearing_vec_camera

            for idx in range(1, len(track.camera_ids)):
                clone = self.state.clones[track.camera_ids[idx]]
                trans = clone.camera_pose_global.t - first_camera_pose.camera_pose_global.t
                parallel_trans = trans.T @ bearing_in_global
                ortho_trans = trans - parallel_trans @ bearing_in_global
                if np.linalg.norm(ortho_trans) > 0.05:
                    return True
            return False

    def update(self, ids):
        """Starts the update process given a list of features.

        Args:
            ids: List of track ids we want to use for the msckf update.

        The main purpose of this function is to validate the tracks for the measurement update.
        """
        if len(ids) == 0:
            return
        map_good_track_id_to_triangulated_pt = {}
        min_track_removed = 0
        bad_motion_removed = 0
        bad_triangulation = 0
        for id in ids:
            track = self.map_id_to_feature_tracks[id]

            # Check if the track has enough keypoints to justify the expense of an update.
            if len(track.tracked_keypoints) < self.params.min_track_length_for_update:
                min_track_removed += 1
                continue

            if not self.check_feature_motion(id):
                bad_motion_removed += 1
                continue
            camera_pose_world_list = []
            # Landmark is good so triangulate it.
            for cam_id in track.camera_ids:
                clone = self.state.clones[cam_id]
                camera_pose_world_list.append(clone.camera_pose_global)

            is_valid, triangulated_pt = linear_triangulate(camera_pose_world_list, track.tracked_keypoints)

            if not is_valid:
                bad_triangulation += 1
                continue

            is_valid, optimized_pt = optimize_point_location(triangulated_pt, camera_pose_world_list,
                                                             track.tracked_keypoints)

            if not is_valid:
                bad_triangulation += 1
                continue

            map_good_track_id_to_triangulated_pt[id] = optimized_pt

        logger.info("Updating with %i tracks out of %i", len(map_good_track_id_to_triangulated_pt), len(ids))
        logger.info("Removed %i tracks due to track length, and %i due to not enough parallax, %i bad triangulation",
                    min_track_removed, bad_motion_removed, bad_triangulation)

        self.update_with_good_ids(map_good_track_id_to_triangulated_pt)

        for id in ids:
            del self.map_id_to_feature_tracks[id]

    def compute_residual_and_jacobian(self, track, pt_global):
        """
        Compute the jacobian and the residual of a 3D point.
        Args:
            track: Track which contains the measurements, and the associated camera poses
            pt_global: The 3D point in the global frame.

        Returns:


        """
        num_measurements = len(track.tracked_keypoints)

        # Preallocate the our matrices. Note that the number of measurements can end up being smaller
        # if one of the measurements corresponds to a invalid clone(was removed during pruning)
        H_f = np.zeros((2 * num_measurements, 3), dtype=np.float64)
        H_X = np.zeros((2 * num_measurements, self.state.get_state_size()), dtype=np.float64)
        residuals = np.empty((2 * num_measurements, ), dtype=np.float64)

        actual_measurement_count = 0
        for idx in range(num_measurements):
            cam_id = track.camera_ids[idx]
            measurement = track.tracked_keypoints[idx]
            clone = None
            clone_index = None
            # We need to iterate through the OrderedDict rather than use the key as we need to find the
            # index within the state vector
            for index, (key, value) in enumerate(self.state.clones.items()):
                if key == cam_id:
                    clone = value
                    clone_index = index
            # Clone doesn't exist/ was removed. Skip this measurement
            if clone_index == None:
                continue
            clone = self.state.clones[cam_id]
            cRw = clone.camera_pose_global.R.T
            cPw = cRw @ -clone.camera_pose_global.t

            pt_camera = cRw @ pt_global + cPw
            # The actual measurement index. Needed if one of the camera clones is invalid.
            m_idx = actual_measurement_count
            # This slice corresponds to the rows that relate to this measurement.
            measurement_slice = slice(2 * m_idx, 2 * m_idx + 2)
            # Compute and set the residuals
            normalized_x = pt_camera[0] / pt_camera[2]
            normalized_y = pt_camera[1] / pt_camera[2]
            error = np.array([measurement[0] - normalized_x, measurement[1] - normalized_y])

            residuals[measurement_slice] = error

            # Compute the jacobian with respect to the feature position.
            # This can be found around Eq. 23 in the tech report
            X = pt_camera[0]
            Y = pt_camera[1]
            invZ = 1.0 / pt_camera[2]
            jac_i = invZ * np.array([[1.0, 0.0, -X * invZ], [0.0, 1.0, -Y * invZ]])

            H_f[measurement_slice] = jac_i @ cRw

            # Compute jacobian with respect to the current camera clone
            temp_mat = np.zeros((3, 4))
            temp_mat[:3, :3] = np.eye(3)
            X_prime, Y_prime, Z_prime = pt_camera[0], pt_camera[1], pt_camera[2]
            uline_l0 = np.array([X_prime, Y_prime, Z_prime, 1])
            dpc0_dxc = -1 * temp_mat @ odotOperator(uline_l0)
            jac_se3 = jac_i @ dpc0_dxc

            # Get the index of the current clone within the state vector. As we need to set their computed jacobians
            clone_state_index = self.state.calc_clone_index(clone_index)

            se3_start_index = clone_state_index + StateInfo.CLONE_SE3_SLICE.start
            se3_end_index = clone_state_index + StateInfo.CLONE_SE3_SLICE.stop
            H_X[measurement_slice, se3_start_index:se3_end_index] = jac_se3

            actual_measurement_count += 1

        if actual_measurement_count != num_measurements:
            assert (False)

        return actual_measurement_count, residuals, H_X, H_f

    def project_left_nullspace(self, matrix):
        """Figure out the left nullspace of the matrix.

        Args:
            matrix: Matrix we want to compute the left nullspace of.

        Returns:
            Matrix representing the left nullspace of the input matrix.

        Linear Algebra recap:
            * The nullspace or kernel is the set of solutions that map to the zero vector.

             matrix * nullspace = 0.

            * The left nullspace or cokernel is the solution that will map to the zero vector if multiplied from the
            left side.

            left_nullspace * matrix = 0

            It can be found by finding the nullspace of the matrix transposed.

            left_nullspace = nullspace(matrix^T)
        """
        A = scipy.linalg.null_space(matrix.T)
        return A

    def chi_square_test(self, H_o, residual, dof):
        noise = np.eye(H_o.shape[0]) * self.params.keypoint_noise_sigma**2
        innovation = H_o @ self.state.covariance @ H_o.T + noise
        # gamma = r^T * (H*P*H^T + R)^-1 * r
        # (H*P*H^T + R)^-1 * r = np.linalg.solve((H*P*H^T + R), r)
        gamma = residual.T @ np.linalg.solve(innovation, residual)
        if gamma < self.chi_square_val[dof]:
            return True
        return False

    def update_with_good_ids(self, map_good_track_ids_to_point_3d):
        """ Run an EKF update with valid tracks.

        Args:
            map_good_track_ids_to_point_3d: Dict which maps track ids to its triangulated point. Should only contain
                tracks that have gone through some sort of preprocessing to remove outliers.

        This function computes the individual jacobians and residuals for each track and combines them into 2 large
        matrices for a big EKF Update at the end.

        It is in this function we use the so called MSCKF update, or nullspace projection which is the big innovation
        introduced in A.I. Mourikis, S.I. Roumeliotis: "A Multi-state Constraint Kalman Filter for Vision-Aided
        Inertial Navigation", and can be found on Equations 23,24.

        The nullspace projection allows us to remove

        """

        num_landmarks = len(map_good_track_ids_to_point_3d)

        if num_landmarks == 0:
            return

        # Here we preallocate our update matrices. This way we can do 1 big EKF update
        # rather then many small ones(is much more efficient).
        # This is the maximum size possible of our update matrix. Each feature provides 2
        # residuals per keypoint * the maximum number of keypoints(max_track_length). The -3
        # comes from the nullspace projection which is explained below.
        max_possible_size = num_landmarks * 2 * self.params.max_track_length - 3
        H = np.empty((max_possible_size, self.state.get_state_size()))
        r = np.zeros((max_possible_size, ))
        index = 0
        for id, triangulated_pt in map_good_track_ids_to_point_3d.items():
            track = self.map_id_to_feature_tracks[id]
            actual_num_measurements, residuals, H_X, H_f = self.compute_residual_and_jacobian(track, triangulated_pt)

            # Nullspace projection.
            A = self.project_left_nullspace(H_f)

            H_o = A.T @ H_X

            rows, cols = H_o.shape
            assert (rows == 2 * actual_num_measurements - 3)
            assert (cols == H_X.shape[1])

            r_o = A.T @ residuals

            dof = residuals.shape[0] / 2 - 1
            if not self.chi_square_test(H_o, r_o, dof):
                continue

            num_residuals = residuals.shape[0]
            start_row = index
            end = index + num_residuals - 3
            r[start_row:end] = r_o
            H[start_row:end] = H_o
            index += num_residuals - 3

        if index == 0:
            return
        final_r = r[0:index]
        final_H = H[0:index]

        R = np.zeros((final_r.shape[0], final_r.shape[0]))
        np.fill_diagonal(R, self.params.keypoint_noise_sigma**2)
        self.update_EKF(final_r, final_H, R)

    def integrate(self, dt, omega, acc):
        R = self.state.imu_R_global
        p = self.state.global_t_imu
        v = self.state.velocity

        # update position 
        Hl = Hl_operator(dt*omega)
        p_new = p + dt*v + self.gravity*((dt**2)/2) + R @ Hl @ acc * (dt**2)

        # update velocity 
        Jl = Jl_operator(dt*omega)
        v_new = v + self.gravity*dt + R @ Jl @ acc * dt

        # update rotation
        delta_R = sp.SO3.exp(dt*omega).matrix()
        R_new = R @ delta_R

        self.state.imu_R_global = R_new
        self.state.global_t_imu = p_new
        self.state.velocity = v_new

    def propagate(self, imu_buffer):
        for imu in imu_buffer:
            # prepare the basic terms 
            wRi = self.state.imu_R_global
            dt = imu.time_interval
            acc_hat = imu.angular_vel - self.state.bias_gyro
            gyro_hat = imu.linear_acc - self.state.bias_acc

            # predict the mean 
            self.integrate(dt, gyro_hat, acc_hat)

            a_skew = skew(acc_hat)
            g_skew = skew(gyro_hat)
            g_norm = np.linalg.norm(gyro_hat)

            # make sure acc, gyro have proper sizes 
            acc_hat = np.reshape(acc_hat, (3, 1))
            gyro_hat = np.reshape(gyro_hat, (3, 1))

            theta_theta = sp.SO3.exp(-dt * gyro_hat).matrix()
            JL_plus = Jl_operator(dt * gyro_hat)
            JL_minus = Jl_operator(-dt * gyro_hat)
            I3 = np.eye(3)
            Delta = -(g_skew / (g_norm**2)) @ (theta_theta.T @ (dt * g_skew - I3) + I3)
            HL_plus = Hl_operator(dt * gyro_hat)
            HL_minus = Hl_operator(-dt * gyro_hat)

            # prepare the terms in Phi 
            theta_gyro = -dt * JL_minus
            v_theta = -dt * wRi @ skew(JL_plus @ acc_hat)
            v_gyro = wRi @ Delta @ a_skew @ (I3 + (g_skew @ g_skew / (g_norm**2))) + dt * wRi @ JL_plus @ (a_skew @ g_skew / (g_norm**2)) + dt * wRi @ (gyro_hat @ acc_hat.T / (g_norm**2)) @ JL_minus - dt * ((acc_hat.T @ gyro_hat).item() / (g_norm**2)) * I3
            v_acc = -dt * wRi @ JL_plus

            p_theta = -(dt**2) * wRi @ skew(HL_plus @ acc_hat)
            p_v = dt * I3
            p_gyro = wRi @ (-g_skew @ Delta - dt * JL_plus + dt * I3) @ a_skew @ (I3 + (g_skew @ g_skew / (g_norm**2))) @ (g_skew / (g_norm**2)) + (dt**2) * wRi @ HL_plus @ (a_skew @ g_skew / (g_norm**2)) + (dt**2) * wRi @ (gyro_hat @ acc_hat.T / (g_norm**2)) @ HL_minus - (dt**2) * ((acc_hat.T @ gyro_hat).item() / (2*(g_norm**2))) * wRi
            p_acc = -(dt**2) * wRi @ HL_plus

            # for Phi 
            Phi = np.eye(15)
            # theta row 
            Phi[0:3, 0:3] = theta_theta
            Phi[0:3, 9:12] = theta_gyro
            # v row 
            Phi[3:6, 0:3] = v_theta
            Phi[3:6, 9:12] = v_gyro
            Phi[3:6, 12:15] = v_acc
            # p row 
            Phi[6:9, 0:3] = p_theta
            Phi[6:9, 3:6] = p_v
            Phi[6:9, 9:12] = p_gyro
            Phi[6:9, 12:15] = p_acc

            # obtain Q 
            Q = Phi @ self.noise_matrix @ Phi.T * dt

            self.state_server.state_cov[0:StateInfo.IMU_STATE_SIZE, 0:StateInfo.IMU_STATE_SIZE] = (
                Phi @ self.state_server.state_cov[0:StateInfo.IMU_STATE_SIZE, 0:StateInfo.IMU_STATE_SIZE] @ Phi.T + Q)

            # Update the imu-camera covariance
            self.state.covariance[0:15, 15:] = (Phi @ self.state.covariance[0:15, 15:])
            self.state.covariance[15:, :15] = (self.state.covariance[15:, :15] @ Phi.T)

            new_cov_symmetric = symmeterize_matrix(self.state.covariance)
            self.state.covariance[0:StateInfo.IMU_STATE_SIZE, 0:StateInfo.IMU_STATE_SIZE] = new_cov_symmetric

    def update_EKF(self, res, H, R):
        """

        Args:
            res:
            H:
            R:

        Returns:

        """
        assert (R.shape[0] == res.shape[0])
        assert (H.shape[0] == res.shape[0])
        logger.info("Residual norm is %f", np.linalg.norm(res))

        if H.shape[0] > H.shape[1] and self.params.use_QR_compression:
            # See "A Multi-state Constraint Kalman Filter for Vision-Aided Inertial Navigation" Eq 26-28 and
            # "An Estimation Algorithm for  Vision-Based Exploration of Small Bodies in Space" Section V Part A.
            #
            # The math has to do with some matrix algebra taking advantage of the fact that Q is an orthogonal matrix
            # and thus its transpose is equal to its inverse. I also advise you to understand what the QR factorization
            # actually does.
            #
            # As for why we do it. Imagine we have 10 features seen in 10 cameras. This results in a matrix with 170
            # rows(nullspace projection reduces it from 200). It can be compressed to a matrix of state vector size
            # so in this example it would be 75(IMU state + 6*10 cameras).
            # The Kalman gain requires matrix inversion which generally is an O(n^3) algorithm. So 75^3<<170^3
            # is a big computation saving.
            # Note that I ignore the extra cost of the QR decomposition, but it ends up being worth it.

            # Here RT is the upper triangular matrix
            Q1, RT = np.linalg.qr(H, mode='reduced')
            H_thin = RT
            r_thin = Q1.T @ res
            R_thin = Q1.T @ R @ Q1
        else:
            H_thin = H
            r_thin = res
            R_thin = R

        H = H_thin
        res = r_thin
        R = R_thin
        H_T = H.transpose()

        cur_cov = self.state.covariance
        K = cur_cov @ H_T @ np.linalg.inv((H @ cur_cov @ H_T + R))
        state_size = self.state.get_state_size()

        # Update the covariance using the joseph form(is more numerically stable)
        new_cov = (np.eye(state_size) - K @ H) @ cur_cov @ (np.eye(state_size) - K @ H).T + K @ R @ K.T
        delta_x = K @ res

        # Apply the new covariance and the update
        self.state.covariance = new_cov
        self.state.update_state(delta_x)

    def augment_camera_state(self, timestamp):
        iRw = self.state.imu_R_global.T
        cRi = self.camera_calib.imu_R_camera.T

        # Compute the pose of the camera in the global frame
        cRw = cRi @ iRw

        wPi = self.state.global_t_imu
        iPc = self.camera_calib.imu_t_camera
        wRi = self.state.imu_R_global
        wPc = wPi + wRi @ iPc

        cur_state_size = self.state.get_state_size()
        # This jacobian stores the partial derivatives of the camera position(6 states) with respect to the current
        # state vector
        jac = np.zeros((StateInfo.CLONE_STATE_SIZE, cur_state_size), dtype=np.float64)
        dcampose_dimupose = get_cam_wrt_imu_se3_jacobian(cRi, iPc, cRw)
        jac[0:3, 0:3] = dcampose_dimupose[0:3, 0:3]
        jac[0:3, 6:9] = dcampose_dimupose[0:3, 3:6]
        jac[3:6, 0:3] = dcampose_dimupose[3:6, 0:3]
        jac[3:6, 6:9] = dcampose_dimupose[3:6, 3:6]

        new_state_size = cur_state_size + StateInfo.CLONE_STATE_SIZE
        # See Eq 15 from above reference
        augmentation_matrix = np.eye(new_state_size, cur_state_size)
        augmentation_matrix[cur_state_size:, :] = jac

        new_covariance = augmentation_matrix @ self.state.covariance @ augmentation_matrix.transpose()
        # Helps with numerical problems
        new_cov_sym = symmeterize_matrix(new_covariance)

        # Add the camera clone and set the new covariance matrix which includes it
        self.camera_id += 1
        clone = CameraClone(cRw.T, wPc, self.camera_id)
        self.state.add_clone(clone)
        self.state.covariance = new_cov_sym

        return self.camera_id


