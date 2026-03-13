# stereo_frame_capture.py
# StereoFrameCapture class

import cv2
import queue
from threading import Thread
from config import CAMERA_LEFT_ID, CAMERA_RIGHT_ID, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

class StereoFrameCapture:
    """Captures synchronized frames from two cameras in background threads."""

    def __init__(self, left_id, right_id, width, height, fps):
        self.left_id  = left_id
        self.right_id = right_id
        self.width    = width
        self.height   = height
        self.fps      = fps
        self.cap_left  = None
        self.cap_right = None
        self.q_left    = queue.Queue(maxsize=5)
        self.q_right   = queue.Queue(maxsize=5)
        self.running   = False

    def start(self):
        self.cap_left  = cv2.VideoCapture(self.left_id)
        self.cap_right = cv2.VideoCapture(self.right_id)
        for cap in [self.cap_left, self.cap_right]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS,          self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        self.running = True
        Thread(target=self._run, args=(self.cap_left,  self.q_left),  daemon=True).start()
        Thread(target=self._run, args=(self.cap_right, self.q_right), daemon=True).start()
        print('✓ Stereo capture started')

    def _run(self, cap, q):
        while self.running:
            ret, frame = cap.read()
            if ret:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                q.put(frame)

    def get_frame_pair(self, timeout=1.0):
        try:
            return self.q_left.get(timeout=timeout), self.q_right.get(timeout=timeout)
        except queue.Empty:
            return None, None

    def stop(self):
        self.running = False
        if self.cap_left:  self.cap_left.release()
        if self.cap_right: self.cap_right.release()
        print('[OK] Stereo capture stopped')

print('[OK] StereoFrameCapture class defined')