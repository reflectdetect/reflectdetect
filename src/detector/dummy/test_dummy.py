from src.detector.dummy.DummyDetector import DummyDetector


def test_dummy_detection():
    detector = DummyDetector()
    assert detector.detect("") == "detector/dummy/dummy_detection.json"
