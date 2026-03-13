# main.py
# Entry point for the hand gesture robot control system

import os
import time
import cv2
import numpy as np
import json
from config import CALIBRATION_FILE, IP_ADDRESS, PORT, CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
from single_camera_calibration import SingleCameraCalibration
from robot_position_detector import RobotPositionDetector
from robot_manager import RobotManager
from tracking import run_single_camera_tracking_mode

# ─────────────────────────────────────────────────────────────────────────
# STEP 1 — CAMERA CALIBRATION
# ─────────────────────────────────────────────────────────────────────────
print('=' * 60)
print('STEP 1 — SINGLE-CAMERA CALIBRATION')
print('=' * 60)

camera_calib = SingleCameraCalibration()
calibration_loaded = camera_calib.load(CALIBRATION_FILE)

if calibration_loaded:
    print(f'✓ Found existing calibration file: {CALIBRATION_FILE}')
    choice = input('Use existing calibration? (y = use it / n = recalibrate): ').strip().lower()
    if choice != 'y':
        calibration_loaded = False
        print('→ Will run new calibration.')
    else:
        print('→ Using existing calibration.')
else:
    print(f'⚠  No calibration file found ({CALIBRATION_FILE}).')
    print('→ Will run new calibration.')

if not calibration_loaded:
    print('\nStarting calibration wizard...')
    if camera_calib.calibrate_checkerboard():
        camera_calib.save(CALIBRATION_FILE)
        print('✓ Calibration complete and saved.')
    else:
        print('✗ Calibration failed. Cannot continue.')
        print('Re-run this cell to try calibration again.')
        calibration_loaded = False

print()

if not calibration_loaded:
    print('⛔ Calibration not complete — re-run this cell before continuing.')
else:
    pass  # calibration done, continue

# ─────────────────────────────────────────────────────────────────────────
# STEP 2 — ROBOT BASE POSE DETECTION (ArUco or Manual)
# ─────────────────────────────────────────────────────────────────────────
print('=' * 60)
print('STEP 2 — ROBOT BASE POSE DETECTION')
print('=' * 60)

robot_detector = RobotPositionDetector(
    camera_matrix=camera_calib.camera_matrix,
    dist_coeffs=camera_calib.dist_coeffs
)

# Check for existing base pose
base_pose_file = 'base_pose.json'
base_pose_loaded = False
robot_base_position = None

if os.path.exists(base_pose_file):
    try:
        with open(base_pose_file, 'r') as f:
            data = json.load(f)
        robot_base_position = np.array(data['robot_base_position'])
        print(f'✓ Base pose loaded from {base_pose_file}: {robot_base_position}')
        base_pose_loaded = True
    except Exception as e:
        print(f'⚠ Error loading base pose: {e}')

if not base_pose_loaded:
    print('Choose base pose detection method:')
    print('1. ArUco marker detection (automatic)')
    print('2. Manual input')
    choice = input('Enter choice (1 or 2): ').strip()

    if choice == '1':
        # ArUco detection
        cap = cv2.VideoCapture(CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        print('Looking for ArUco markers on robot base...')
        print('Press ENTER when robot is detected, or q to skip.')

        detected = False
        while True:
            ret, frame = cap.read()
            if not ret:
                print('⚠ Camera read failed.')
                break

            _, annotated = robot_detector.detect(frame)

            if robot_detector.is_detected:
                x, y, z = robot_detector.robot_position
                banner = f'ROBOT DETECTED: ({x:.3f}, {y:.3f}, {z:.3f}) m — press ENTER to confirm'
                color = (0, 255, 0)
                detected = True
            else:
                banner = 'Robot NOT detected — ensure markers face camera'
                color = (0, 0, 255)

            cv2.putText(annotated, banner, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.imshow('Base Pose Detection', annotated)

            key = cv2.waitKey(30) & 0xFF
            if key == 13 and detected:  # Enter
                robot_base_position = robot_detector.robot_position.copy()
                print(f'✓ Robot base position confirmed: {robot_base_position}')
                # Save to JSON
                data = {'robot_base_position': robot_base_position.tolist()}
                with open(base_pose_file, 'w') as f:
                    json.dump(data, f, indent=4)
                print(f'✓ Base pose saved to {base_pose_file}')
                break
            elif key == ord('q'):
                print('⚠ Skipped ArUco detection.')
                break

        cap.release()
        cv2.destroyAllWindows()

    elif choice == '2':
        # Manual input
        print('Enter robot base position manually (in meters):')
        try:
            x = float(input('X coordinate: '))
            y = float(input('Y coordinate: '))
            z = float(input('Z coordinate: '))
            robot_base_position = np.array([x, y, z])
            print(f'✓ Manual base position set: {robot_base_position}')
            # Save to JSON
            data = {'robot_base_position': robot_base_position.tolist()}
            with open(base_pose_file, 'w') as f:
                json.dump(data, f, indent=4)
            print(f'✓ Base pose saved to {base_pose_file}')
        except ValueError:
            print('⚠ Invalid input. Using default base position.')
            robot_base_position = np.array([0.0, 0.0, 1.5])

    else:
        print('⚠ Invalid choice. Using default base position.')
        robot_base_position = np.array([0.0, 0.0, 1.5])

if robot_base_position is None:
    print('⚠ No base position set. Using default.')
    robot_base_position = np.array([0.0, 0.0, 1.5])

print(f'→ Robot base position: {robot_base_position}')
print()

# ─────────────────────────────────────────────────────────────────────────
# STEP 3 — ROBOT CONNECTION (optional — tracking works without it)
# ─────────────────────────────────────────────────────────────────────────
print('=' * 60)
print('STEP 3 — ROBOT CONNECTION')
print('=' * 60)

robot_manager = RobotManager(IP_ADDRESS, PORT)

try:
    connection_ok = robot_manager.connect()
except Exception as e:
    print(f'⚠  Connection error: {e}')
    connection_ok = False

if connection_ok:
    print('✓ Robot connected and ready.')
else:
    print('⚠  Could not connect to robot at', IP_ADDRESS)
    print('   Tracking and ArUco detection will still run.')
    print('   Robot movements will be SKIPPED until connection is established.')

print()

# ─────────────────────────────────────────────────────────────────────────
# STEP 4 — START TRACKING
# ─────────────────────────────────────────────────────────────────────────
print('=' * 60)
print('STEP 4 — STARTING TRACKING LOOP')
print('=' * 60)

try:
    run_single_camera_tracking_mode(camera_calib, robot_manager, robot_detector, robot_base_position)
except KeyboardInterrupt:
    print('\nInterrupted by user.')
finally:
    robot_manager.disconnect()
    print('Done.')