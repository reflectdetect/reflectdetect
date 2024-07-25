from pathlib import Path

from src.detector.naive.NaiveDetector import NaiveDetector


def local_test_naive_detector():
    detector = NaiveDetector("C:/Users/ninja/reflectdetect/data/example/panel_data.json")
    detector.detect("C:/Users/ninja/reflectdetect/data/example/IMG_0040_1.tif")
    detector.detect("C:/Users/ninja/reflectdetect/data/example/IMG_0040_2.tif")
    detector.detect("C:/Users/ninja/reflectdetect/data/example/IMG_0040_3.tif")
    detector.detect("C:/Users/ninja/reflectdetect/data/example/IMG_0040_4.tif")
    detector.detect("C:/Users/ninja/reflectdetect/data/example/IMG_0040_5.tif")


def test_naive_detector():
    detector = NaiveDetector("C:/Users/ninja/reflectdetect/data/example/panel_data.json")
    detector.detect(
        (Path.cwd() / "data/example/YOLO_OBB_Dataset/Images/seq1/seq1_00040_01.jpg").resolve().absolute().as_posix())
