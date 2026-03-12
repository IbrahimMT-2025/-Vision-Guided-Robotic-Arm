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
L1 = 0.252   # Base  → Shoulder
L2 = 0.237   # Shoulder → Elbow
L3 = 0.420   # Elbow → End-effector

# ── Single camera setup ───────────────────────────────────────────────────────
CAMERA_ID        = 0
CAMERA_WIDTH     = 1280
CAMERA_HEIGHT    = 720
CAMERA_FPS       = 60

# ── Fallback robot base offset (used ONLY if ArUco markers are not visible) ──
# When ArUco is working these values are ignored.
ROBOT_BASE_X = 0.0   # +X = robot is to the RIGHT of camera center
ROBOT_BASE_Y = 0.0   # +Y = robot is BELOW camera center
ROBOT_BASE_Z = 1.5   # +Z = robot is further AWAY from camera

# ── State machine timing ──────────────────────────────────────────────────────
HAND_HOLD_TIME               = 5.0   # seconds hand must be held steady
POSITION_STABILITY_THRESHOLD = 0.02  # meters
COOLDOWN_TIME                = 5.0   # seconds after move completes

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