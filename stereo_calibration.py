# stereo_calibration.py
# StereoCalibration class

import cv2
import numpy as np
import json
import os
from config import CALIBRATION_FILE, CAMERA_LEFT_ID, CAMERA_RIGHT_ID, DISPARITY_NUM_DISPARITIES, DISPARITY_BLOCK_SIZE, STEREO_BASELINE

class StereoCalibration:
    """Handles stereo camera calibration, rectification, and 3-D triangulation."""

    def __init__(self):
        self.camera_matrix_left  = None
        self.camera_matrix_right = None
        self.dist_coeffs_left    = None
        self.dist_coeffs_right   = None
        self.rotation_matrix     = None
        self.translation_vector  = None
        self.essential_matrix    = None
        self.fundamental_matrix  = None
        self.rect_transform_left  = None
        self.rect_transform_right = None
        self.projection_matrix_left  = None
        self.projection_matrix_right = None
        self.disparity_matrix   = None
        self.roi_left           = None
        self.roi_right          = None
        self.rectify_map_left   = None
        self.rectify_map_right  = None
        self.stereo_matcher     = None
        self.is_calibrated      = False
        self.image_size         = None

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filename):
        if not self.is_calibrated:
            print('⚠  Calibration not complete — nothing saved')
            return False
        try:
            data = {
                'camera_matrix_left':  self.camera_matrix_left.tolist(),
                'camera_matrix_right': self.camera_matrix_right.tolist(),
                'dist_coeffs_left':    self.dist_coeffs_left.tolist(),
                'dist_coeffs_right':   self.dist_coeffs_right.tolist(),
                'rotation_matrix':     self.rotation_matrix.tolist(),
                'translation_vector':  self.translation_vector.tolist(),
                'image_size':          list(self.image_size),
                'baseline':            STEREO_BASELINE,
            }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            print(f'✓ Calibration saved → {filename}')
            return True
        except Exception as e:
            print(f'⚠  Save error: {e}')
            return False

    def load(self, filename):
        if not os.path.exists(filename):
            return False
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.camera_matrix_left  = np.array(data['camera_matrix_left'])
            self.camera_matrix_right = np.array(data['camera_matrix_right'])
            self.dist_coeffs_left    = np.array(data['dist_coeffs_left'])
            self.dist_coeffs_right   = np.array(data['dist_coeffs_right'])
            self.rotation_matrix     = np.array(data['rotation_matrix'])
            self.translation_vector  = np.array(data['translation_vector'])
            self.image_size          = tuple(data['image_size'])
            self._compute_rectification()
            self.is_calibrated = True
            print(f'✓ Calibration loaded ← {filename}')
            return True
        except Exception as e:
            print(f'⚠  Load error: {e}')
            return False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_rectification(self):
        if self.camera_matrix_left is None:
            return False
        try:
            ret = cv2.stereoRectify(
                self.camera_matrix_left,  self.dist_coeffs_left,
                self.camera_matrix_right, self.dist_coeffs_right,
                self.image_size,
                self.rotation_matrix, self.translation_vector,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0
            )
            (self.rect_transform_left, self.rect_transform_right,
             self.projection_matrix_left, self.projection_matrix_right,
             self.disparity_matrix, self.roi_left, self.roi_right) = ret

            self.rectify_map_left = cv2.initUndistortRectifyMap(
                self.camera_matrix_left,  self.dist_coeffs_left,
                self.rect_transform_left, self.projection_matrix_left,
                self.image_size, cv2.CV_32F
            )
            self.rectify_map_right = cv2.initUndistortRectifyMap(
                self.camera_matrix_right,  self.dist_coeffs_right,
                self.rect_transform_right, self.projection_matrix_right,
                self.image_size, cv2.CV_32F
            )
            self.stereo_matcher = cv2.StereoBM_create(
                numDisparities=DISPARITY_NUM_DISPARITIES,
                blockSize=DISPARITY_BLOCK_SIZE
            )
            return True
        except Exception as e:
            print(f'⚠  Rectification error: {e}')
            return False

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate_stereo_automatic(self):
        """Interactive checkerboard stereo calibration."""
        print('\n=== STEREO CALIBRATION ===')
        print('Checkerboard: 7x5 inner corners, square size 3cm')
        print('c = capture (needs both cameras to see board) | s = calibrate | q = quit\n')

        # Use LOW resolution for calibration — faster, more reliable detection
        # (same as the working single-camera test)
        CALIB_W, CALIB_H = 640, 480

        cap_left  = cv2.VideoCapture(CAMERA_LEFT_ID)
        cap_right = cv2.VideoCapture(CAMERA_RIGHT_ID)

        for cap in [cap_left, cap_right]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CALIB_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CALIB_H)
            cap.set(cv2.CAP_PROP_FPS,          30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # flush stale frames

        # Check cameras actually opened
        if not cap_left.isOpened():
            print(f'✗ Cannot open left camera (ID={CAMERA_LEFT_ID})')
            return False
        if not cap_right.isOpened():
            print(f'✗ Cannot open right camera (ID={CAMERA_RIGHT_ID})')
            cap_left.release()
            return False

        import time
        time.sleep(1)  # let cameras initialise before reading

        cb_size  = (7, 5)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp     = np.zeros((np.prod(cb_size), 3), np.float32)
        objp[:, :2] = np.mgrid[0:cb_size[0], 0:cb_size[1]].T.reshape(-1, 2)
        objp *= 0.03

        objpoints, imgpoints_left, imgpoints_right = [], [], []
        image_count = 0

        print('Cameras ready. Show checkerboard to both cameras.')

        while True:
            ret_l, frame_left  = cap_left.read()
            ret_r, frame_right = cap_right.read()

            if not ret_l or not ret_r:
                print('⚠  Camera read failed — check connections')
                break

            gray_l = cv2.cvtColor(frame_left,  cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

            ok_l, corners_l = cv2.findChessboardCorners(gray_l, cb_size, None)
            ok_r, corners_r = cv2.findChessboardCorners(gray_r, cb_size, None)

            disp_l = frame_left.copy()
            disp_r = frame_right.copy()

            if ok_l:
                corners_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), criteria)
                cv2.drawChessboardCorners(disp_l, cb_size, corners_l, ok_l)
            if ok_r:
                corners_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), criteria)
                cv2.drawChessboardCorners(disp_r, cb_size, corners_r, ok_r)

            # Per-camera status — same style as working single-cam test
            l_status = "DETECTED" if ok_l else "NOT DETECTED"
            r_status = "DETECTED" if ok_r else "NOT DETECTED"
            l_color  = (0, 255, 0) if ok_l else (0, 0, 255)
            r_color  = (0, 255, 0) if ok_r else (0, 0, 255)

            cv2.putText(disp_l, f'LEFT:  {l_status}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, l_color, 2)
            cv2.putText(disp_l, f'Captured: {image_count}/20', (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(disp_l, 'c=capture (need BOTH)  s=calibrate  q=quit', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.putText(disp_r, f'RIGHT: {r_status}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, r_color, 2)
            cv2.putText(disp_r, f'Captured: {image_count}/20', (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('LEFT Camera  — Calibration', disp_l)
            cv2.imshow('RIGHT Camera — Calibration', disp_r)

            key = cv2.waitKey(30) & 0xFF   # 30ms — same as working single-cam test

            if key == ord('q'):
                print('Cancelled.')
                break

            elif key == ord('c'):
                if ok_l and ok_r:
                    objpoints.append(objp)
                    imgpoints_left.append(corners_l)
                    imgpoints_right.append(corners_r)
                    image_count += 1
                    print(f'  ✓ Captured frame {image_count}')
                else:
                    missing = []
                    if not ok_l: missing.append('LEFT')
                    if not ok_r: missing.append('RIGHT')
                    print(f'  ⚠  Board not seen by: {", ".join(missing)} — try again')

            elif key == ord('s'):
                if image_count < 5:
                    print(f'  Need at least 5 frames (have {image_count})')
                    continue

                print(f'\nCalibrating with {image_count} frames...')
                try:
                    img_size = gray_l.shape[::-1]

                    _, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
                        objpoints, imgpoints_left,  img_size, None, None)
                    _, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
                        objpoints, imgpoints_right, img_size, None, None)

                    ok_s, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                        objpoints, imgpoints_left, imgpoints_right,
                        mtx_l, dist_l, mtx_r, dist_r,
                        img_size,
                        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
                        flags=cv2.CALIB_FIX_INTRINSIC
                    )

                    if ok_s:
                        self.camera_matrix_left  = mtx_l
                        self.camera_matrix_right = mtx_r
                        self.dist_coeffs_left    = dist_l
                        self.dist_coeffs_right   = dist_r
                        self.rotation_matrix     = R
                        self.translation_vector  = T
                        self.essential_matrix    = E
                        self.fundamental_matrix  = F
                        self.image_size          = img_size
                        self._compute_rectification()
                        self.is_calibrated = True
                        print('✓ Stereo calibration successful!')
                        break
                    else:
                        print('⚠  Calibration failed — capture more frames at different angles')

                except Exception as e:
                    print(f'⚠  Calibration error: {e}')

        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()
        return self.is_calibrated


    # ── 3-D geometry ──────────────────────────────────────────────────────────

    def triangulate(self, point_left, point_right):
        """Return 3-D world position (meters) from two corresponding 2-D pixel coords."""
        if not self.is_calibrated:
            return None
        try:
            pt_l = cv2.undistortPoints(
                np.array([[[point_left[0],  point_left[1]]],  ], dtype=np.float32),
                self.camera_matrix_left,  self.dist_coeffs_left,
                P=self.projection_matrix_left
            )[0][0]
            pt_r = cv2.undistortPoints(
                np.array([[[point_right[0], point_right[1]]], ], dtype=np.float32),
                self.camera_matrix_right, self.dist_coeffs_right,
                P=self.projection_matrix_right
            )[0][0]
            p4d = cv2.triangulatePoints(
                self.projection_matrix_left,  self.projection_matrix_right,
                pt_l.reshape(2,1), pt_r.reshape(2,1)
            )
            return (p4d[:3] / p4d[3]).flatten()
        except Exception as e:
            print(f'⚠  Triangulation error: {e}')
            return None

    def compute_disparity_depth(self, frame_left, frame_right):
        """Fast depth map via stereo block matching (optional visualisation)."""
        if not self.is_calibrated or self.stereo_matcher is None:
            return None, None
        try:
            rl = cv2.remap(frame_left,  *self.rectify_map_left,  cv2.INTER_LINEAR)
            rr = cv2.remap(frame_right, *self.rectify_map_right, cv2.INTER_LINEAR)
            gl = cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)
            disp  = self.stereo_matcher.compute(gl, gr).astype(np.float32) / 16.0
            focal = self.projection_matrix_left[0, 0]
            depth = (STEREO_BASELINE * focal) / (disp + 1e-5)
            return disp, depth
        except Exception as e:
            print(f'⚠  Disparity error: {e}')
            return None, None

print('✓ StereoCalibration class defined')