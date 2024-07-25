import os
from pathlib import Path

from src.extractor.naive.Extractor import Extractor


def test_naive_extractor():
    extractor = Extractor("reflectance_panel_example_data.json")
    try:
        assert (extractor.extract("data/example/YOLO_OBB_Dataset/Images/seq1/seq1_00040_00.jpg",
                                  "src/detector/dummy/dummy_detection.json"
                                  )
                == (
                            Path.cwd() / "data/example/YOLO_OBB_Dataset/Images/seq1/metadata/seq1_00040_00_extraction.json").resolve())
    finally:
        path = "data/example/YOLO_OBB_Dataset/Images/seq1/metadata/seq1_00040_00_extraction.json"
        if os.path.exists(path):
            os.remove(path)

