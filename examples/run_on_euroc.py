import logging
import os
from multiprocessing import Process, Queue

import click
import cv2
import numpy as np

from src.dataset_utils import TimestampSynchronizer, csv_read_matrix
from src.feature_tracker import FeatureTracker
from src.orcvio import ORCVIO
from src.orcvio_types import CameraCalibration, IMUData, PinholeIntrinsics
from src.params import AlgorithmConfig, EurocDatasetCalibrationParams
from src.spatial_transformations import hamiltonian_quaternion_to_rot_matrix
from src.viewer import create_and_run

logger = logging.getLogger(__name__)

LEFT_CAMERA_FOLDER = "cam0"
RIGHT_CAMERA_FOLDER = "cam1"
IMU_FOLDER = "imu0"
GT_FOLDER = "state_groundtruth_estimate0"
DATA_FILE = "data.csv"
# RESULT_FILE = "./cache/msckf/"
RESULT_FILE = "./cache/orcvio/"

TIMESTAMP_INDEX = 0
NANOSECOND_TO_SECOND = 1e-9
# Euroc fastest running sensor is the IMU at about 200 hz or 0.005 seconds.(Not exact). This is the actual value in
# nanoseconds.
EUROC_DELTA_TIME = 5000192

levels = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warn': logging.WARNING,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}


@click.command()
@click.option('--euroc_folder', required=False, default="/mnt/disk2/euroc/MH_02_easy/mav0", help="Path to a folder containing Euroc data. Typically called mav0")
@click.option('--start_timestamp', required=False, default="1403636896901666560", help="Timestamp of where we want to start reading data from.")
@click.option('--use_viewer', is_flag=True, help="Use a 3D viwer to view the camera path")
@click.option('--log_level', required=False, default="debug", help="Level of python logging messages")
def run_on_euroc(euroc_folder, start_timestamp, use_viewer, log_level):

    level = levels.get(log_level.lower())
    logging.basicConfig(format='%(filename)s: %(message)s', level=level)

    euroc_calib = EurocDatasetCalibrationParams()
    camera_calib = CameraCalibration()
    camera_calib.intrinsics = PinholeIntrinsics.initialize(euroc_calib.cam0_intrinsics,
                                                           euroc_calib.cam0_distortion_model,
                                                           euroc_calib.cam0_distortion_coeffs)
    camera_calib.set_extrinsics(euroc_calib.T_imu_cam0)
    config = AlgorithmConfig()
    feature_tracker = FeatureTracker(config.feature_tracker_params, camera_calib)

    orcvio = ORCVIO(config.orcvio_params, camera_calib)
    orcvio.set_imu_noise(0.005, 0.05, 0.001, 0.01)
    orcvio.set_imu_covariance(1e-5, 1e-2, 1e-12, 1e-2, 1e-2)

    imu_data = csv_read_matrix(os.path.join(euroc_folder, IMU_FOLDER, DATA_FILE))
    camera_data = csv_read_matrix(os.path.join(euroc_folder, LEFT_CAMERA_FOLDER, DATA_FILE))
    imu_timestamps = [int(data[0]) for data in imu_data]
    camera_timestamps = [int(data[0]) for data in camera_data]
    ground_truth_data = csv_read_matrix(os.path.join(euroc_folder, GT_FOLDER, DATA_FILE))
    ground_truth_timestamps = [int(data[0]) for data in ground_truth_data]

    time_syncer = TimestampSynchronizer(int(EUROC_DELTA_TIME / 2))

    time_syncer.add_timestamp_stream("camera", camera_timestamps)
    time_syncer.add_timestamp_stream("imu", imu_timestamps)
    time_syncer.add_timestamp_stream("gt", ground_truth_timestamps)
    time_syncer.set_start_timestamp(int(start_timestamp))
    imu_buffer = []
    last_imu_timestamp = -1
    first_time = True

    est_pose_queue = None
    ground_truth_queue = None
    if use_viewer:
        est_pose_queue = Queue()
        ground_truth_queue = Queue()
        viewer_process = Process(target=create_and_run, args=(est_pose_queue, ground_truth_queue))
        viewer_process.start()
    # for covariance analysis 
    covariance_queue = []

    while time_syncer.has_data():

        cur_data = time_syncer.get_data()
        if "imu" in cur_data:
            imu_index = cur_data["imu"].index
            imu_line = imu_data[imu_index]
            measurements = np.array([imu_line[1:]]).astype(np.float64).squeeze()

            gyro = np.array(measurements[0:3])
            acc = np.array(measurements[3:])
            timestamp = int(imu_line[TIMESTAMP_INDEX])
            if last_imu_timestamp != -1:
                dt = timestamp - last_imu_timestamp
            else:
                dt = EUROC_DELTA_TIME
            last_imu_timestamp = timestamp
            dt_seconds = dt * NANOSECOND_TO_SECOND
            timestamp_seconds = timestamp * NANOSECOND_TO_SECOND
            imu_buffer.append(IMUData(acc, gyro, timestamp_seconds, dt_seconds))

        if "camera" in cur_data:
            index = cur_data["camera"].index
            image_name = camera_data[index][1]
            img = cv2.imread(os.path.join(euroc_folder, LEFT_CAMERA_FOLDER, "data", image_name), 0)

            feature_tracker.track(img, imu_buffer)
            measurements, ids = feature_tracker.get_current_normalized_keypoints_and_ids()
            if first_time:
                gt_index = cur_data["gt"].index
                gt_line = ground_truth_data[gt_index]
                gt = np.array([gt_line[1:]]).astype(np.float64).squeeze()
                gt_pos = gt[0:3]
                gt_quat = gt[3:7]
                gt_vel = gt[7:10]
                gt_bias_gyro = gt[10:13]
                gt_bias_acc = gt[13:16]
                gt_rot_matrx = hamiltonian_quaternion_to_rot_matrix(gt_quat)
                orcvio.initialize(gt_rot_matrx, gt_pos, gt_vel, gt_bias_acc, gt_bias_gyro)
                first_time = False
                continue

            orcvio.propagate(imu_buffer)
            orcvio.add_camera_features(ids, measurements)
            est_rot_mat = orcvio.state.imu_R_global
            est_trans = orcvio.state.global_t_imu
            est_pose = np.eye(4, dtype=np.float32)
            est_pose[0:3, 0:3] = est_rot_mat
            est_pose[0:3, 3] = est_trans
            if est_pose_queue:
                est_pose_queue.put(est_pose)
            # for covariance analysis 
            covariance_queue.append(orcvio.state.get_pos_covariance())
            orcvio.remove_old_clones()
            imu_buffer.clear()

            if ground_truth_queue and "gt" in cur_data:
                gt_index = cur_data["gt"].index
                gt_line = ground_truth_data[gt_index]
                gt = np.array([gt_line[1:]]).astype(np.float64).squeeze()
                gt_pos = gt[0:3]
                gt_quat = gt[3:7]
                gt_rot_mat = hamiltonian_quaternion_to_rot_matrix(gt_quat)
                gt_transform = np.eye(4, dtype=np.float32)
                gt_transform[0:3, 0:3] = gt_rot_mat
                gt_transform[0:3, 3] = gt_pos
                ground_truth_queue.put(gt_transform)

    # for covariance analysis 
    np.savez(RESULT_FILE + 'covariances.npz', covariance_queue)

if __name__ == '__main__':
    run_on_euroc()
