import json
import os
from pathlib import Path

from src.transformer.naive.Transformer import Transformer


def test_naive_transformer():
    transformer = Transformer()
    with open("temp_extraction.json", "w") as f:
        example_extraction_data = [[35891.487282411515, 0.65], [64823.316067775675, 0.67]]
        json.dump(example_extraction_data, f)
    try:
        assert (transformer.transform("data/example/IMG_0040_1.tif",
                                      "temp_extraction.json")
                == (Path.cwd() / "data/example/reflectance/IMG_0040_1_transformed.png").resolve())
    finally:
        os.remove("temp_extraction.json")
        os.remove("data/example/reflectance/IMG_0040_1_transformed.png")
