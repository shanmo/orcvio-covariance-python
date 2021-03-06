from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np


class ViewerConfig():
    """Stores the parameters for our Visualization/Viewer."""
    class WindowTypes():
        """Different ways to initialize the OpenGL window.

        We utilize the moderngl_window class to handle our OpenGL context and other stuff such as keyboard bindings,
        mouse control,...

        moderngl_window allows for multiple different ways to initialize the OpenGL context. These are the possible
        options.
        """
        default = None
        pyglet = "pyglet"
        pygame = "pygame2"
        glfw = "glfw"
        sdl2 = "sdl2"
        pyside2 = "pyside2"
        pyqt5 = "pyqt5"
        tk = "tk"

    def __init__(self):
        # Param which controls which library we use to initialize the OpenGL window. Must be from 'WindowTypes'
        self.window_type = ViewerConfig.WindowTypes.default
        self.vsync = True
        # Size of the Viewer window
        self.size = (1920, 1080)
        # Is the window resizeable
        self.resizable = True
        # Set the window to fullscreen.
        self.fullscreen = False
        # Show the cursor in the OpenGL window
        self.show_cursor = True


class AlgorithmConfig():
    @dataclass
    class FeatureTrackerParams():
        @dataclass
        class OpticalFlowTrackerParams():
            window_size = 15
            use_iteration_stopping_criteria = True
            use_minimum_error_stopping_criteria = True
            max_iterations = 30
            minimum_error = 1e-3
            max_pyramid_level = 3

            def to_opencv_dict(self):
                stopping_criteria = 0
                if self.use_iteration_stopping_criteria:
                    stopping_criteria = stopping_criteria | cv2.TERM_CRITERIA_COUNT
                if self.use_minimum_error_stopping_criteria:
                    stopping_criteria = stopping_criteria | cv2.TERM_CRITERIA_EPS

                return dict(winSize=(self.window_size, self.window_size),
                            maxLevel=self.max_pyramid_level,
                            criteria=(stopping_criteria, self.max_iterations, self.minimum_error))

        @dataclass()
        class KeyPointDetectorParams():

            # Maximum new corners to detect in detector.
            max_corners = 200
            quality_level = 0.01
            # A new point must be at least this distance away from another point to be considered valid.
            min_distance = 10.0
            # The size of the window used to look at to deteremine the sub-pixel location of the keypoint.
            sub_pix_window_size = 10
            sub_pix_zero_zone = -1

        lk_params: OpticalFlowTrackerParams = OpticalFlowTrackerParams()
        detector_params: KeyPointDetectorParams = KeyPointDetectorParams()

        # Set this to a value to make the run deterministic. Is used to set the randomness for the RANSAC algorithm.
        numpy_random_seed = None

        small_angle_threshold = 0.001745329

        block_size = 200

        ransac_iterations = 300
        ransac_threshold = 1e-4

        # If the number of tracked features falls below this value, then it will try to detect new keypoints.
        min_tracked_features = 200

        # The maximum amount of features we will track.
        max_tracked_features = 200

        # The length and width of a cell within our grid.
        grid_block_size = 100
        # Maximum keypoints we allow per cell
        max_keypoints_per_block = 10
        # Within a cell we require a new keypoint to have atleast this much distance from older keypoints.
        min_dist_between_keypoints = 10

    @dataclass
    class ORCVIOParams():
        # The maximum number of clones a track can have, before it must be used for an update.
        max_track_length = 20

        # The minimum track length a track must have to justify a MSCKF update with it.
        min_track_length_for_update = 3

        # use_observability_constraint = False

        # How noisy we believe a keypoint measurement is.
        keypoint_noise_sigma = 0.35

        # The maximum number of clones we store. Anymore more than this and we start deleting older clones.
        max_clone_num = 20

        # In the EKF update it is sometimes possible to compress the matrices to a smaller size. This is useful as it
        # it makes subsequent operations such as computing the Kalman Gain much more efficient.
        use_QR_compression = True

    feature_tracker_params: FeatureTrackerParams = FeatureTrackerParams()
    orcvio_params: ORCVIOParams = ORCVIOParams()


@dataclass()
class EurocDatasetCalibrationParams():
    T_imu_cam0 = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                           [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                           [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                           [0, 0, 0, 1.000000000000000]])
    cam0_distortion_model = 'radtan'
    cam0_distortion_coeffs = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])
    cam0_intrinsics = np.array([458.654, 457.296, 367.215, 248.375])
    cam0_resolution = np.array([752, 480])
