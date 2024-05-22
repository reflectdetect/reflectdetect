import json
import os
from pathlib import Path

from src.transformer.naive_batch.BatchTransformer import BatchTransformer


def test_naive_batch_transformer():
    transformer = BatchTransformer()
    with open("temp_extraction.json", "w") as f:
        example_extraction_data = [[[35891.487282411515, 0.65], [64823.316067775675, 0.67]]]
        json.dump(example_extraction_data, f)
    try:
        assert (transformer.transform(["data/example/YOLO_OBB_Dataset/Images/seq1/seq1_00040_00.jpg"],
                                      "temp_extraction.json")
                == (Path.cwd() / "data/example/YOLO_OBB_Dataset/Images/seq1/reflectance").resolve())
    finally:

        os.remove("temp_extraction.json")
        path = "data/example/YOLO_OBB_Dataset/Images/seq1/reflectance/seq1_00040_00_transformed.png"
        if os.path.exists(path):
            os.remove(path)
