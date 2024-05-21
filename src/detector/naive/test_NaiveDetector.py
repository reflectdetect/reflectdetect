from src.detector.naive.NaiveDetector import NaiveDetector


def test_naive_detector():
    detector = NaiveDetector("C:/Users/ninja/reflectdetect/data/example/panel_data.json")
    detector.detect("C:/Users/ninja/reflectdetect/data/example/IMG_0040_1.tif")
    detector.detect("C:/Users/ninja/reflectdetect/data/example/IMG_0040_2.tif")
    detector.detect("C:/Users/ninja/reflectdetect/data/example/IMG_0040_3.tif")
    detector.detect("C:/Users/ninja/reflectdetect/data/example/IMG_0040_4.tif")
    detector.detect("C:/Users/ninja/reflectdetect/data/example/IMG_0040_5.tif")