import json
import os
from pathlib import Path

from src.extractor.naive_batch.BatchExtractor import BatchExtractor


def test_naive_batch_extractor():
    with open("reflectance_panel_example_data.json") as f:
        panel_data = json.load(f)
    extractor = BatchExtractor(panel_data)
    try:
        assert (extractor.extract(["data/example/YOLO_OBB_Dataset/Images/seq1/seq1_00040_00.jpg",
                                   "data/example/YOLO_OBB_Dataset/Images/seq1/seq1_00040_01.jpg"],
                                  ["src/detector/dummy/dummy_detection.json",
                                   "src/detector/dummy/dummy_detection.json", ]
                                  )
                == (
                        Path.cwd() / "data/example/YOLO_OBB_Dataset/Images/seq1/metadata/40_extraction.json").resolve())
    finally:
        path = "data/example/YOLO_OBB_Dataset/Images/seq1/metadata/40_extraction.json"
        if os.path.exists(path):
            os.remove(path)
