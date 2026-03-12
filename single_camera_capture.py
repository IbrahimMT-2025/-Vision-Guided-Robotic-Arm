# single_camera_capture.py
# Simple wrapper for grabbing frames from a single camera asynchronously.

import cv2
import queue
from threading import Thread

class SingleCameraCapture:
    """Background thread grabbing latest frame from one camera."""

    def __init__(self, cam_id, width, height, fps):
        self.cam_id = cam_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.queue = queue.Queue(maxsize=1)
        self.running = False

    def start(self):
        self.cap = cv2.VideoCapture(self.cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True
        Thread(target=self._run, daemon=True).start()
        print('✓ Single-camera capture started')

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
                self.queue.put(frame)

    def get_frame(self, timeout=1.0):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        print('✓ Single-camera capture stopped')