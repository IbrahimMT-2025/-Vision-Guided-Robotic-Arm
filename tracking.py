# tracking.py
# Main tracking function (single camera version)

import cv2
import time
import math
from cvzone.HandTrackingModule import HandDetector
from enums import SystemState
from robot_state_controller import RobotStateController
from single_camera_capture import SingleCameraCapture

# Try to use cvzone HandDetector, fallback to custom implementation if cvzone fails
try:
    detector = HandDetector(detectionCon=0.8, maxHands=1)
    USE_CUSTOM_DETECTOR = False
except AttributeError:
    print("⚠ cvzone HandDetector failed, using basic fallback detector")
    import mediapipe as mp

    class BasicHandDetector:
        def __init__(self, detectionCon=0.8, maxHands=1):
            print("🔧 Using basic hand detection (motion-based)")
            self.detection_confidence = detectionCon
            self.prev_frame = None
            self.frame_count = 0
            self.hand_present_frames = 0

        def findHands(self, img, draw=True):
            self.frame_count += 1

            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            hands = []

            if self.prev_frame is not None:
                # Calculate frame difference
                frame_delta = cv2.absdiff(self.prev_frame, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)

                # Find contours
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                motion_detected = False
                for contour in contours:
                    if cv2.contourArea(contour) > 3000:  # Minimum area for hand-sized motion
                        motion_detected = True

                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)

                        # Create hand-like structure for compatibility
                        hand = {
                            'lmList': [
                                [0, x + w//2, y + h//2],  # Center point (like fingertip)
                                [1, x + w//4, y + h//4],  # Top-left
                                [2, x + 3*w//4, y + h//4],  # Top-right
                                [3, x + w//4, y + 3*h//4],  # Bottom-left
                                [4, x + 3*w//4, y + 3*h//4],  # Bottom-right
                                [5, x + w//2, y + h//4],  # Top-center
                                [6, x + w//4, y + h//2],  # Left-center
                                [7, x + 3*w//4, y + h//2],  # Right-center
                                [8, x + w//2, y + 3*h//4],  # Bottom-center
                            ],
                            'bbox': [x, y, w, h],
                            'center': [x + w//2, y + h//2],
                            'type': 'Motion'
                        }
                        hands.append(hand)

                        if draw:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.circle(img, (x + w//2, y + h//2), 5, (0, 0, 255), -1)

                        break  # Only return one hand

                if motion_detected:
                    self.hand_present_frames += 1
                else:
                    self.hand_present_frames = max(0, self.hand_present_frames - 1)

                # Only consider hand present if motion detected in recent frames
                if self.hand_present_frames < 3:
                    hands = []

            self.prev_frame = gray.copy()
            return hands, img

    USE_CUSTOM_DETECTOR = True
    CustomHandDetector = BasicHandDetector
from config import CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, HAND_HOLD_TIME, ROBOT_BASE_X, ROBOT_BASE_Y, ROBOT_BASE_Z


def run_single_camera_tracking_mode(calib, robot_manager, robot_detector=None, robot_base_position=None):
    """Real-time hand tracking using a single camera plus optional ArUco robot pose.

    Depth (Z) is estimated from the pixel width of the hand using a simple
    pinhole-camera model.  X/Y are computed by unprojecting the fingertip
    coordinate using the calibrated intrinsics and the estimated Z.
    """
    if not calib.is_calibrated:
        print('⚠  Camera calibration not complete — run calibration first')
        return

    print('\n=== TRACKING MODE (SINGLE CAMERA + ARUCO) ===')
    print('Hand tracking active. Press q to quit.\n')

    frame_capture = SingleCameraCapture(CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
    frame_capture.start()
    time.sleep(2)  # let camera warm up

    # Use the appropriate detector
    if USE_CUSTOM_DETECTOR:
        detector = BasicHandDetector(detectionCon=0.8, maxHands=1)
    else:
        detector = HandDetector(detectionCon=0.8, maxHands=1)

    state_controller = RobotStateController(robot_manager, robot_detector, robot_base_position)

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

                    # estimate depth using hand width (landmarks 0 and 5)
                    pixel_width = int(math.dist(lm_list[0], lm_list[5]))
                    world_z = calib.estimate_depth(pixel_width)

                    if world_z is not None and world_z > 0:
                        # unproject X/Y
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
                            # Use the provided robot_base_position
                            if robot_base_position is not None:
                                rel_x = world_x - robot_base_position[0]
                                rel_y = world_y - robot_base_position[1]
                                rel_z = world_z - robot_base_position[2]
                                cv2.putText(frame,
                                            f'Robot: ({rel_x:.3f}, {rel_y:.3f}, {rel_z:.3f}) [base pose]',
                                            (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
                            else:
                                cv2.putText(frame,
                                            f'Robot: ({world_x-ROBOT_BASE_X:.3f}, {world_y-ROBOT_BASE_Y:.3f}, {world_z-ROBOT_BASE_Z:.3f}) [fixed]',
                                            (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

                except Exception as e:
                    print(f'⚠ Tracking error: {e}')

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