# main.py
# Entry point for the hand gesture robot control system

import sys
import os
import io

# Set UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time
import cv2
from config import CALIBRATION_FILE, IP_ADDRESS, PORT, CAMERA_LEFT_ID, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
from stereo_calibration import StereoCalibration
from robot_position_detector import RobotPositionDetector
from robot_manager import RobotManager
from tracking import run_stereo_tracking_mode

# ─────────────────────────────────────────────────────────────────────────
# STEP 1 — STEREO CALIBRATION
# ─────────────────────────────────────────────────────────────────────────
print('=' * 60)
print('STEP 1 — STEREO CALIBRATION')
print('=' * 60)

stereo_calib = StereoCalibration()
print(f'Working dir: {os.getcwd()}')
print(f'Looking for calibration file at: {CALIBRATION_FILE} (exists? {os.path.exists(CALIBRATION_FILE)})')
calibration_loaded = stereo_calib.load(CALIBRATION_FILE)

# allow automated scripts to bypass the prompt by setting AUTO_CALIB=1
auto_accept = os.environ.get('AUTO_CALIB', '') == '1'

if calibration_loaded:
    print(f'✓ Found existing calibration file: {CALIBRATION_FILE}')
    if auto_accept or not sys.stdin.isatty():
        # non‑interactive environment or user opted in
        print('→ Using existing calibration (auto)')
    else:
        choice = input('Use existing calibration? (y = use it / n = recalibrate): ').strip().lower()
        if choice != 'y':
            calibration_loaded = False
            print('→ Will run new calibration.')
        else:
            print('→ Using existing calibration.')

if not calibration_loaded:
    print('\nStarting stereo calibration wizard...')
    if stereo_calib.calibrate_stereo_automatic():
        stereo_calib.save(CALIBRATION_FILE)
        print('✓ Calibration complete and saved.')
    else:
        print('✗ Calibration failed. Cannot continue.')
        print('Re-run this cell to try calibration again.')
        calibration_loaded = False

print()

if not calibration_loaded:
    print('⛔ Calibration not complete — re-run this cell before continuing.')
    sys.exit(1)  # Exit if calibration failed

# double‑check that the loaded data actually contains a usable left camera matrix
if stereo_calib.camera_matrix_left is None or stereo_calib.camera_matrix_left.shape != (3, 3):
    print('⛔ Loaded calibration is invalid — cannot run robot detection.')
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────
# STEP 2 — DETECT ROBOT VIA ARUCO MARKERS  (only runs if calibration_loaded)
# ─────────────────────────────────────────────────────────────────────────
print('=' * 60)
print('STEP 2 — ROBOT DETECTION (ArUco markers)')
print('=' * 60)
print('Opening left camera to look for ArUco markers on robot base...')
print('Press any key to capture a frame for detection.')
print()

robot_detector = RobotPositionDetector(
    camera_matrix=stereo_calib.camera_matrix_left,
    dist_coeffs=stereo_calib.dist_coeffs_left
)

cap_check = cv2.VideoCapture(CAMERA_LEFT_ID)
cap_check.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
cap_check.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap_check.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

robot_found_at_startup = False

print('Looking for robot markers (press q to skip)...')
while True:
    ret, frame = cap_check.read()
    if not ret:
        print('⚠  Could not read from camera.')
        break

    robot_pos, annotated = robot_detector.detect(frame)

    # Status banner
    if robot_detector.is_detected:
        x, y, z = robot_detector.robot_position
        banner = f'ROBOT DETECTED  ({x:.3f}, {y:.3f}, {z:.3f}) m  — press ENTER to continue'
        color  = (0, 255, 0)
        robot_found_at_startup = True
    else:
        banner = 'Robot NOT detected — make sure markers face the camera  |  q = skip'
        color  = (0, 0, 255)

    cv2.putText(annotated, banner, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.imshow('STEP 2 — Robot Detection', annotated)

    key = cv2.waitKey(30) & 0xFF
    if key == 13 and robot_detector.is_detected:   # Enter
        print(f'✓ Robot base detected at: {robot_detector.robot_position}')
        break
    elif key == ord('q'):
        print('⚠  Skipped — tracking will use fixed ROBOT_BASE_X/Y/Z offset.')
        robot_detector = None
        break

cap_check.release()
cv2.destroyAllWindows()
time.sleep(3)  # let camera hardware fully release before tracking re-opens it

if robot_found_at_startup:
    print('→ Robot position confirmed. Coordinate conversion will use live ArUco tracking.')
else:
    print('→ Robot not visible yet. System will keep checking during tracking loop.')

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
    run_stereo_tracking_mode(stereo_calib, robot_manager, robot_detector)
except KeyboardInterrupt:
    print('\nInterrupted by user.')
finally:
    robot_manager.disconnect()
    print('Done.')