import cv2
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal

class CameraThread(QThread):
    frame_signal = pyqtSignal(QImage)

    def __init__(self, camera_index=1):
        super().__init__()
        self.running = True
        self.camera_index = 1

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        while self.running:
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_qt_format.scaled(600, 510)
                self.frame_signal.emit(p)

        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
