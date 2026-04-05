# config.py
# Configuration constants and imports

import cv2
import cv2.aruco as aruco
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# cri_lib is optional; robot movement will be disabled if it's unavailable
try:
    from cri_lib import CRIController
except ImportError:
    CRIController = None
    print('⚠ cri_lib not installed; robot connection disabled')

import time
import math
import json
import os
from enum import Enum
from collections import deque
from threading import Thread
import queue

print('=' * 70)
print('HAND GESTURE ROBOT CONTROL - SINGLE CAMERA + ARUCO VERSION')
print('=' * 70)

# ── Robot connection ──────────────────────────────────────────────────────────
IP_ADDRESS       = '192.168.3.11'
PORT             = 3920
CALIBRATION_FILE = 'single_camera_calibration.json'

# ── Robot arm link lengths (meters, from manual) ─────────────────────────────
#  Base(L1) → Shoulder(L2) → Elbow(L3 spans A4 roll-joint) → Wrist(L4) → EE
L1 = 0.252   # Base  → Shoulder (A1-turn)
L2 = 0.237   # Shoulder → Elbow (A2-pivot)
L3 = 0.297   # Elbow → Wrist (spans A4 turning joint)
L4 = 0.126   # Wrist → End-effector (A5-pivot, A6-turn)

# ── Single camera setup ───────────────────────────────────────────────────────
CAMERA_ID              = 0
CAMERA_WIDTH           = 1280
CAMERA_HEIGHT          = 720
CAMERA_FPS             = 60
FOCAL_LENGTH           = 850          # pixels (override by calibration file if present)
INDEX_FINGER_WIDTH     = 0.020        # metres (MCP→PIP physical width; tune if needed)
CAMERA_CALIBRATION_FILE = 'camera_calibration.json'

# ── Fallback robot base offset (used ONLY if ArUco markers are not visible) ──
# When ArUco is working these values are ignored.
ROBOT_BASE_X = 0.0   # +X = robot is to the RIGHT of camera center
ROBOT_BASE_Y = 0.0   # +Y = robot is BELOW camera center
ROBOT_BASE_Z = 1.5   # +Z = robot is further AWAY from camera

# ── State machine timing ──────────────────────────────────────────────────────
HAND_HOLD_TIME               = 5.0   # seconds hand must be held steady
POSITION_STABILITY_THRESHOLD = 0.015 # meters (1.5 cm dead zone)
COOLDOWN_TIME                = 5.0   # seconds after move completes

# ── Joint limits (degrees) ───────────────────────────────────────────────────
JOINT_LIMITS = {
    'A1': (-170, 170),
    'A2': (-135,  90),
    'A3': (-120, 120),
    'A4': (-170, 170),
    'A5': (-135, 135),
    'A6': (-170, 170),
}

# ── Hand tracking sensitivity ────────────────────────────────────────────────
XY_SENSITIVITY = 1.5   # amplify X and Y hand movement
Z_SENSITIVITY  = 2.0   # amplify Z (depth) hand movement
Z_SMOOTH_FACTOR = 0.6  # exponential smoothing for depth: 0 = very smooth, 1 = no smoothing

# ── Workspace safety clamp (robot frame, metres) ─────────────────────────────
RZ_MIN = -1.3   # furthest forward  (1.3 m in front of robot, toward camera)
RZ_MAX = 0.4    # furthest back
RY_MIN = -0.4   # below shoulder (table clearance)
RY_MAX = 0.7    # max height above base

# ── Disparity/depth settings (not used in single camera) ─────────────────────
USE_DISPARITY_CALCULATION  = False
DISPARITY_NUM_DISPARITIES  = 16 * 5
DISPARITY_BLOCK_SIZE       = 15

# ── ArUco marker settings ────────────────────────────────────────────────────
MARKER_SIZE_METERS = 0.10   # Physical printed marker size (10 cm)

# Where each marker ID sits relative to the robot BASE CENTER (x, y, z) meters
# Measure these after printing and attaching markers to the robot base.
MARKER_OFFSETS = {
    0: np.array([ 0.0,   0.0,  0.15]),  # Front face
    1: np.array([ 0.15,  0.0,  0.0 ]),  # Right face
    2: np.array([ 0.0,   0.0, -0.15]),  # Back face
    3: np.array([-0.15,  0.0,  0.0 ]),  # Left face
}

print('✓ Configuration loaded')