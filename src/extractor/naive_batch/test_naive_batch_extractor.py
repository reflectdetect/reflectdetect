import json
import os
from pathlib import Path

from src.extractor.naive_batch.BatchExtractor import BatchExtractor


def test_naive_batch_extractor():
    with open("reflectance_panel_example_data.json") as f:
        panel_data = json.load(f)
    extractor = BatchExtractor(panel_data)
    try:
        assert (extractor.extract(["data/example/IMG_0040_1.tif", "data/example/IMG_0040_2.tif"],
                                  "dummy_detection.json"
                                  )
                == (Path.cwd() / "data/example/metadata/0040_extraction.json").resolve())
    finally:
        os.remove("data/example/metadata/0040_extraction.json")
