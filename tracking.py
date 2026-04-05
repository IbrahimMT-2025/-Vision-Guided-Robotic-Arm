# tracking.py
# Main tracking function (single camera version)

import cv2
import time
import math
from cvzone.HandTrackingModule import HandDetector
from enums import SystemState
from robot_state_controller import RobotStateController
from single_camera_capture import SingleCameraCapture
from config import (CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, HAND_HOLD_TIME, 
                    ROBOT_BASE_X, ROBOT_BASE_Y, ROBOT_BASE_Z, FOCAL_LENGTH, 
                    INDEX_FINGER_WIDTH, Z_SMOOTH_FACTOR)


# ============================================================================
# DEPTH FILTERING AND ESTIMATION
# ============================================================================

class DepthFilter:
    """Exponential smoothing filter for depth values."""
    def __init__(self, factor=Z_SMOOTH_FACTOR):
        self.factor = factor
        self.z = None

    def filter(self, z):
        """Apply exponential smoothing to depth value."""
        if self.z is None:
            self.z = z
            return z
        self.z = self.factor * z + (1 - self.factor) * self.z
        return self.z

    def reset(self):
        """Reset filter state."""
        self.z = None


def calculate_hand_depth(hand, fx=FOCAL_LENGTH):
    """
    Estimate depth from index-finger MCP(5)→PIP(6) pixel segment width.
    
    Parameters
    ----------
    hand : dict
        Hand detection dict with 'lmList' containing landmarks (from cvzone)
    fx : float
        Calibrated focal length in pixels
    
    Returns
    -------
    float or None
        Estimated depth in metres, or None if calculation fails
    """
    try:
        lm = hand['lmList']
        mcp, pip = lm[5], lm[6]  # Index finger MCP and PIP joints
        px_w = math.hypot(pip[0] - mcp[0], pip[1] - mcp[1])
        
        if px_w < 5:  # pixel width too small for reliable depth
            return None
        
        DEPTH_SCALE = 1.0
        z = (INDEX_FINGER_WIDTH * fx / px_w) * DEPTH_SCALE
        return max(0.05, min(1.5, z))  # clamp to reasonable range
    except Exception:
        return None


def run_single_camera_tracking_mode(calib, robot_manager, robot_detector=None):
    """
    Real-time hand tracking using a single camera plus optional ArUco robot pose.

    Depth (Z) is estimated from index-finger MCP→PIP pixel segment width.
    X/Y are computed by unprojecting the fingertip coordinate using the 
    calibrated intrinsics and the estimated Z.
    """
    if not calib.is_calibrated:
        print('⚠  Camera calibration not complete — run calibration first')
        return

    print('\n=== TRACKING MODE (SINGLE CAMERA + ARUCO) ===')
    print('Hand tracking active. Press q to quit.\n')

    frame_capture = SingleCameraCapture(CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
    frame_capture.start()
    time.sleep(2)  # let camera warm up

    detector = HandDetector(detectionCon=0.8, maxHands=1)
    state_controller = RobotStateController(robot_manager, robot_detector)
    depth_filter = DepthFilter()

    frame_count = 0
    STATE_COLORS = {
        SystemState.IDLE:            (0, 255, 0),
        SystemState.TRACKING:        (255, 255, 0),
        SystemState.POSITION_LOCKED: (0, 255, 255),
        SystemState.EXECUTING:       (0, 0, 255),
        SystemState.COOLDOWN:        (255, 0, 255),
    }

    try:
        while True:
            frame = frame_capture.get_frame(timeout=0.5)
            if frame is None:
                continue
            frame_count += 1

            # ArUco robot position detection on the current frame
            if robot_detector is not None:
                _, frame = robot_detector.detect(frame)

            hands, frame = detector.findHands(frame)
            hand_detected = False
            world_x = world_y = world_z = None

            if hands:
                try:
                    lm_list = hands[0]['lmList']
                    fingertip = lm_list[8]
                    pixel = (int(fingertip[0]), int(fingertip[1]))

                    # Estimate depth using index finger MCP→PIP width (improved method)
                    z_raw = calculate_hand_depth(hands[0], fx=calib.camera_matrix[0, 0])
                    
                    if z_raw is not None:
                        # Apply smoothing filter
                        world_z = depth_filter.filter(z_raw)

                        # Unproject X/Y using calibrated camera matrix
                        fx = calib.camera_matrix[0, 0]
                        fy = calib.camera_matrix[1, 1]
                        cx = calib.camera_matrix[0, 2]
                        cy = calib.camera_matrix[1, 2]
                        world_x = (pixel[0] - cx) * world_z / fx
                        world_y = (pixel[1] - cy) * world_z / fy
                        hand_detected = True

                        cv2.circle(frame, pixel, 10, (0, 255, 0), -1)
                        cv2.putText(frame,
                                    f'Camera: ({world_x:.3f}, {world_y:.3f}, {world_z:.3f})',
                                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                        if robot_detector is not None and robot_detector.is_detected:
                            rel = robot_detector.get_robot_to_hand_vector((world_x, world_y, world_z))
                            if rel is not None:
                                cv2.putText(frame,
                                            f'Robot:  ({rel[0]:.3f}, {rel[1]:.3f}, {rel[2]:.3f}) [ArUco]',
                                            (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
                        elif robot_detector is not None:
                            cv2.putText(frame,
                                        'WARNING: ArUco not visible',
                                        (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        else:
                            cv2.putText(frame,
                                        f'Robot: ({world_x-ROBOT_BASE_X:.3f}, {world_y-ROBOT_BASE_Y:.3f}, {world_z-ROBOT_BASE_Z:.3f}) [fixed]',
                                        (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
                    else:
                        depth_filter.reset()

                except Exception as e:
                    print(f'⚠ Tracking error: {e}')
                    depth_filter.reset()

            else:
                depth_filter.reset()

            # state machine update
            state = state_controller.update(hand_detected, world_x, world_y, world_z)

            color = STATE_COLORS.get(state, (128,128,128))
            cv2.putText(frame, f'STATE: {state_controller.get_state_name()}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if state == SystemState.TRACKING:
                remaining = max(0, HAND_HOLD_TIME - state_controller.get_time_in_state())
                cv2.putText(frame, f'HOLD: {remaining:.1f}s', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,200,0), 2)

            cv2.putText(frame, f'Frame: {frame_count}  |  q = quit',
                        (10, CAMERA_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            cv2.imshow('Single Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        frame_capture.stop()
        cv2.destroyAllWindows()
        print('✓ Tracking stopped')

print('✓ run_single_camera_tracking_mode defined')
def run_stereo_tracking_mode(stereo_calib, robot_manager, robot_detector=None):
    """
    Real-time stereo hand tracking with ArUco-based robot base detection.

    Flow each frame:
      1. Capture left + right frames.
      2. Detect robot base position via ArUco (left camera only).
      3. Detect hand landmarks in both cameras.
      4. Triangulate fingertip to get 3-D camera-world position.
      5. State machine converts camera coords → robot-relative coords
         using the live ArUco measurement (step 2).
      6. When stable, send IK command to robot.
    """
    if not stereo_calib.is_calibrated:
        print('⚠  Stereo calibration not complete — run Cell 13 first')
        return

    print('\n=== TRACKING MODE (STEREO + ARUCO) ===')
    print('Hand tracking active.  Press q to quit.\n')

    # ── Setup ─────────────────────────────────────────────────────────────────
    frame_capture = StereoFrameCapture(
        CAMERA_LEFT_ID, CAMERA_RIGHT_ID,
        CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
    )
    frame_capture.start()
    time.sleep(2)   # Camera warm-up

    detector = HandDetector(detectionCon=0.8, maxHands=1)

    # ── ArUco robot position detector (uses LEFT camera calibration) ──────────
    # robot_detector is passed in from main() startup sequence
    # Falls back to None if not provided (movements will be blocked)

    # ── State machine gets the robot_detector so it can do live coord conv ────
    state_controller = RobotStateController(robot_manager, robot_detector)

    frame_count  = 0
    STATE_COLORS = {
        SystemState.IDLE:            (0,   255, 0),
        SystemState.TRACKING:        (255, 255, 0),
        SystemState.POSITION_LOCKED: (0,   255, 255),
        SystemState.EXECUTING:       (0,   0,   255),
        SystemState.COOLDOWN:        (255, 0,   255),
    }

    try:
        while True:
            frame_left, frame_right = frame_capture.get_frame_pair(timeout=0.5)
            if frame_left is None:
                continue
            frame_count += 1

            # ── Step 2: Detect robot base via ArUco ──
            if robot_detector is not None:
                _, frame_left = robot_detector.detect(frame_left)

            # ── Step 3: Detect hand landmarks ─────────────────────────────────
            hands_left,  frame_left  = detector.findHands(frame_left)
            hands_right, frame_right = detector.findHands(frame_right)

            hand_detected = False
            world_x = world_y = world_z = None

            if hands_left and hands_right:
                try:
                    lm_l = hands_left[0]['lmList'][8]   # Index fingertip
                    lm_r = hands_right[0]['lmList'][8]
                    px_l = (int(lm_l[0]), int(lm_l[1]))
                    px_r = (int(lm_r[0]), int(lm_r[1]))

                    # ── Step 4: Triangulate ───────────────────────────────────
                    point_3d = stereo_calib.triangulate(px_l, px_r)

                    if point_3d is not None:
                        world_x, world_y, world_z = point_3d
                        hand_detected = True

                        cv2.circle(frame_left,  px_l, 10, (0, 255, 0), -1)
                        cv2.circle(frame_right, px_r, 10, (0, 255, 0), -1)

                        # HUD: camera coords
                        cv2.putText(frame_left,
                                    f'Camera: ({world_x:.3f}, {world_y:.3f}, {world_z:.3f})',
                                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                        # HUD: robot-relative coords
                        if robot_detector is not None and robot_detector.is_detected:
                            rel = robot_detector.get_robot_to_hand_vector(point_3d)
                            if rel is not None:
                                cv2.putText(frame_left,
                                            f'Robot:  ({rel[0]:.3f}, {rel[1]:.3f}, {rel[2]:.3f}) [ArUco]',
                                            (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
                        elif robot_detector is not None:
                            cv2.putText(frame_left,
                                        'WARNING: ArUco not visible',
                                        (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        else:
                            cv2.putText(frame_left,
                                        f'Robot: ({world_x-ROBOT_BASE_X:.3f}, {world_y-ROBOT_BASE_Y:.3f}, {world_z-ROBOT_BASE_Z:.3f}) [fixed]',
                                        (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

                except Exception as e:
                    print(f'⚠  Tracking error: {e}')

            # ── Debug rectification every 100 frames ──────────────────────────
            if frame_count % 100 == 0:
                debug_rectification(stereo_calib, frame_left, frame_right)

            # ── Step 5 & 6: State machine ─────────────────────────────────────
            state = state_controller.update(hand_detected, world_x, world_y, world_z)

            # ── HUD: state ────────────────────────────────────────────────────
            color = STATE_COLORS.get(state, (128,128,128))
            cv2.putText(frame_left, f'STATE: {state_controller.get_state_name()}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if state == SystemState.TRACKING:
                remaining = max(0, HAND_HOLD_TIME - state_controller.get_time_in_state())
                cv2.putText(frame_left, f'HOLD: {remaining:.1f}s',
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,200,0), 2)

            cv2.putText(frame_left, f'Frame: {frame_count}  |  q = quit',
                        (10, CAMERA_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            cv2.imshow('Stereo Left',  frame_left)
            cv2.imshow('Stereo Right', frame_right)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        frame_capture.stop()
        cv2.destroyAllWindows()
        print('✓ Tracking stopped')

print('✓ run_stereo_tracking_mode defined')