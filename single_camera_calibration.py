# single_camera_calibration.py
# Calibration for a single camera using a checkerboard pattern.

import cv2
import numpy as np
import json
import os
from config import CALIBRATION_FILE, CAMERA_ID


class SingleCameraCalibration:
    """Handles single-camera intrinsics and depth estimation."""

    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_size = None
        self.is_calibrated = False

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filename):
        if not self.is_calibrated:
            print('⚠ Calibration not complete — nothing saved')
            return False
        try:
            data = {
                'camera_matrix': self.camera_matrix.tolist(),
                'dist_coeffs': self.dist_coeffs.tolist(),
                'image_size': list(self.image_size)
            }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            print(f'✓ Calibration saved → {filename}')
            return True
        except Exception as e:
            print(f'⚠ Save error: {e}')
            return False

    def load(self, filename):
        if not os.path.exists(filename):
            return False
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.camera_matrix = np.array(data['camera_matrix'])
            self.dist_coeffs = np.array(data['dist_coeffs'])
            self.image_size = tuple(data['image_size'])
            self.is_calibrated = True
            print(f'✓ Calibration loaded ← {filename}')
            return True
        except Exception as e:
            print(f'⚠ Load error: {e}')
            return False

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate_checkerboard(self, checkerboard_size=(7, 5), square_size=0.03):
        """Interactive checkerboard calibration for a single camera."""
        print('\n=== SINGLE-CAMERA CALIBRATION ===')
        print(f'Checkerboard: {checkerboard_size[0]}x{checkerboard_size[1]} inner corners')
        print("c = capture | s = calibrate | q = quit\n")

        cap = cv2.VideoCapture(CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f'✗ Cannot open camera (ID={CAMERA_ID})')
            return False

        import time
        time.sleep(1)

        objp = np.zeros((np.prod(checkerboard_size), 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size

        objpoints = []
        imgpoints = []
        image_count = 0
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        while True:
            ret, frame = cap.read()
            if not ret:
                print('⚠ Camera read failed')
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ok, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            disp = frame.copy()

            if ok:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(disp, checkerboard_size, corners, ok)

            cv2.putText(disp, f'Captured: {image_count}/20', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(disp, 'c=capture  s=calibrate  q=quit', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            cv2.imshow('Calibration', disp)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c') and ok:
                objpoints.append(objp)
                imgpoints.append(corners)
                image_count += 1
                print(f'  ✓ Captured frame {image_count}')
            elif key == ord('s'):
                if image_count < 5:
                    print(f'  Need at least 5 frames (have {image_count})')
                    continue
                print(f'\nCalibrating with {image_count} frames...')
                ret, mtx, dist, _, _ = cv2.calibrateCamera(
                    objpoints, imgpoints, gray.shape[::-1], None, None)
                if ret:
                    self.camera_matrix = mtx
                    self.dist_coeffs = dist
                    self.image_size = gray.shape[::-1]
                    self.is_calibrated = True
                    print('✓ Calibration successful!')
                    break
                else:
                    print('⚠ Calibration failed')

        cap.release()
        cv2.destroyAllWindows()
        return self.is_calibrated

    # ── Depth estimation ──────────────────────────────────────────────────────

    def estimate_depth(self, pixel_size, real_size=0.08):
        """Estimate Z distance from pixel-size of known object (e.g. hand width)."""
        if self.camera_matrix is None:
            return None
        focal = self.camera_matrix[0, 0]
        depth = (real_size * focal) / (pixel_size + 1e-6)
        return depth


print('✓ SingleCameraCalibration class defined')