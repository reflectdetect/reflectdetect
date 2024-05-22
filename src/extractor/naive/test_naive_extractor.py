import json
import os
from pathlib import Path

from src.extractor.naive.Extractor import Extractor


def test_naive_extractor():
    extractor = Extractor("reflectance_panel_example_data.json")
    try:
        assert (extractor.extract("data/example/IMG_0040_1.tif",
                                  "src/detector/dummy/dummy_detection.json"
                                  )
                == (Path.cwd() / "data/example/metadata/IMG_0040_1_extraction.json").resolve())
    finally:
        os.remove("data/example/metadata/IMG_0040_1_extraction.json")
