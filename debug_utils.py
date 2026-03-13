# debug_utils.py
# Debug utility functions

import cv2
import numpy as np

def debug_rectification(stereo_calib, frame_left, frame_right):
    """
    Draw horizontal epipolar lines to verify stereo rectification.
    After good calibration, the same real-world point should appear
    on the SAME horizontal row in both rectified images.
    """
    rl = cv2.remap(frame_left,  *stereo_calib.rectify_map_left,  cv2.INTER_LINEAR)
    rr = cv2.remap(frame_right, *stereo_calib.rectify_map_right, cv2.INTER_LINEAR)
    combined = np.hstack([rl, rr])
    for y in range(0, combined.shape[0], 60):
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)
    cv2.imshow('Rectification Check', combined)

print('[OK] debug_rectification helper defined')