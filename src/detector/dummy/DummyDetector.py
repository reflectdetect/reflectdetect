from src.detector.BaseDetector import BaseDetector


class DummyDetector(BaseDetector):
    def __init__(self):
        pass

    def detect(self, image_path: str):
        return "src/detector/dummy/dummy_detection.json"
