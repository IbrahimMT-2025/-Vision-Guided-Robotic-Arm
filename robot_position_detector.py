# robot_position_detector.py
# RobotPositionDetector class

import cv2
import cv2.aruco as aruco
import numpy as np
from config import MARKER_SIZE_METERS, MARKER_OFFSETS

class RobotPositionDetector:
    """
    Detects robot base position in camera-world coordinates
    using ArUco markers physically attached to the robot base.

    Replaces the fixed ROBOT_BASE_X/Y/Z offsets with live pose estimation.
    Uses the LEFT camera matrix (passed in from StereoCalibration).
    """

    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs   = dist_coeffs

        # ArUco detector (4x4 dictionary — simple and robust at distance)
        aruco_dict         = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        aruco_params       = aruco.DetectorParameters()
        self.detector      = aruco.ArucoDetector(aruco_dict, aruco_params)

        # State
        self.robot_position  = None   # (x, y, z) in camera-world meters
        self.robot_rotation  = None   # 3x3 rotation matrix
        self.is_detected     = False
        self.last_seen_marker = None

        print('✓ RobotPositionDetector ready (ArUco DICT_4X4_50)')

    # ── Core detection ────────────────────────────────────────────────────────

    def detect(self, frame):
        """
        Detect ArUco markers in `frame` and compute robot base position.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from the LEFT stereo camera.

        Returns
        -------
        robot_position : np.ndarray | None
            (x, y, z) robot base in camera coordinates, or None if not visible.
        annotated_frame : np.ndarray
            Frame with marker overlays drawn on it.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        annotated = frame.copy()

        if ids is None:
            self.is_detected = False
            cv2.putText(annotated, 'Robot: NOT DETECTED', (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return None, annotated

        # Draw raw marker outlines
        aruco.drawDetectedMarkers(annotated, corners, ids)

        positions_from_markers = []

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id not in MARKER_OFFSETS:
                continue  # Unknown marker — skip

            # Estimate this marker's pose in camera space
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                corners[i], MARKER_SIZE_METERS,
                self.camera_matrix, self.dist_coeffs
            )
            rvec = rvec[0][0]
            tvec = tvec[0][0]   # marker origin in camera coords

            # Draw 3-axis frame on marker for visual debug
            cv2.drawFrameAxes(annotated, self.camera_matrix, self.dist_coeffs,
                              rvec, tvec, MARKER_SIZE_METERS * 0.5)

            R, _ = cv2.Rodrigues(rvec)

            # Back-calculate robot base center:
            #   robot_base = marker_pos_in_camera - R * marker_offset_from_base
            offset       = MARKER_OFFSETS[marker_id]
            robot_base   = tvec - R @ offset

            positions_from_markers.append({
                'marker_id':     marker_id,
                'robot_position': robot_base,
                'rotation':      R,
            })
            self.last_seen_marker = marker_id

            # HUD: individual marker distance
            cv2.putText(annotated,
                        f'M{marker_id}: ({tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f})m',
                        (10, 200 + marker_id * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if not positions_from_markers:
            self.is_detected = False
            return None, annotated

        # Average over all visible markers → more markers = higher accuracy
        avg_pos = np.mean(
            [p['robot_position'] for p in positions_from_markers], axis=0)

        self.robot_position = avg_pos
        self.robot_rotation = positions_from_markers[0]['rotation']
        self.is_detected    = True

        x, y, z = avg_pos
        cv2.putText(annotated,
                    f'ROBOT BASE: ({x:.3f}, {y:.3f}, {z:.3f})m',
                    (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return avg_pos, annotated

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def get_robot_to_hand_vector(self, hand_position):
        """
        Convert hand position from camera-world coords to robot-relative coords.

        Parameters
        ----------
        hand_position : array-like, shape (3,)
            Hand position (x, y, z) in camera-world coordinates
            (output of stereo triangulation).

        Returns
        -------
        np.ndarray | None
            (x, y, z) of hand relative to robot base, ready for IK.
            Returns None if robot base is not currently detected.
        """
        if not self.is_detected or self.robot_position is None:
            return None
        return np.array(hand_position) - self.robot_position

print('✓ RobotPositionDetector class defined')