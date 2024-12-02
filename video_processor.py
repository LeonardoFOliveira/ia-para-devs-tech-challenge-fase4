import cv2

class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Não foi possível abrir o vídeo: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_number = 0

    def get_next_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_number += 1
            return frame
        else:
            return None

    def release(self):
        self.cap.release()