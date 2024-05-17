import json
import os
from pathlib import Path

from src.extractor.naive.Extractor import Extractor


def test_naive_extractor():
    with open("reflectance_panel_example_data.json") as f:
        panel_data = json.load(f)
    extractor = Extractor(panel_data)
    try:
        assert (extractor.extract("data/example/IMG_0040_1.tif",
                                  "src/detector/dummy/dummy_detection.json"
                                  )
                == (Path.cwd() / "data/example/metadata/IMG_0040_1_extraction.json").resolve())
    finally:
        os.remove("data/example/metadata/IMG_0040_1_extraction.json")
