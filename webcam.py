from cv2 import VideoCapture

class Webcam:
    def __init__(self, url):
        self.video = VideoCapture(url)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.video.isOpened():
            raise StopIteration
        
        ret = False
        while not ret:
            ret, frame = self.video.read()
        
        return frame
    
    def close(self):
        self.video.release()
