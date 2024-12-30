import cv2
import threading
import time
import logging


class CameraManager:
    def __init__(self, source):
        """
        Initialize the CameraManager object.

        :param source: The video source (e.g., local camera index or URL).
        """
        self.source = source
        self.cap = None
        self.running:bool = False
        self.latest_frame = None
        self.lock = threading.Lock()

    def is_running(self):
        """
        Check if the camera feed is running.

        :return: True if running, False otherwise.
        """
        return self.running

    def start(self):
        """
        Start the camera feed in a background thread.
        """
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.thread.start()
            logging.info("CameraManager background thread started.")

    def stop(self):
        """
        Stop the camera feed and release resources.
        """
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logging.info("CameraManager stopped and resources released.")

    def _capture_frames(self):
        """
        Background thread to capture frames from the camera.
        """
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                logging.info(f"Opening video source: {self.source}")
                self.cap = cv2.VideoCapture(self.source)

            if self.cap.isOpened():
                success, frame = self.cap.read()
                if success:
                    with self.lock:
                        self.latest_frame = frame
                else:
                    logging.error("Failed to read frame. Reopening camera.")
                    self.cap.release()
                    self.cap = None
            else:
                logging.error("Failed to open video source. Retrying in 5 seconds.")
                time.sleep(5)

    def get_frame(self):
        """
        Get the latest frame captured by the camera.

        :return: The latest frame, or None if unavailable.
        """
        with self.lock:
            return self.latest_frame

    def get_fps(self):
        """
        Get the FPS of the camera.

        :return: FPS value, or 0 if unavailable.
        """
        if self.cap is not None and self.cap.isOpened():
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0

    def get_resolution(self):
        """
        Get the resolution of the camera feed.

        :return: A tuple (width, height), or (0, 0) if unavailable.
        """
        if self.cap is not None and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)
    
    def get(self, action=cv2.CAP_PROP_FRAME_WIDTH):
        return self.cap.get(action)