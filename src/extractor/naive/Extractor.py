import json
import os
from pathlib import Path

import cv2
import numpy as np

from src.extractor.BaseExtractor import BaseExtractor
from src.utils.panel import get_panel_factors_for_band
from src.extractor.shared.shared import get_mean_radiance_values
from src.utils.paths import get_image_band, get_extraction_path


class Extractor(BaseExtractor):

    def get_name(self) -> str:
        return "naive"

    def __init__(self, panel_data_path: str):
        # Load panel data
        with open(panel_data_path) as f:
            self.panel_data = json.load(f)

    def extract(self, image_path: str, detection_path: str) -> str:
        # get band identifier from image path
        band = get_image_band(image_path)

        # Load detection results
        with open(detection_path) as f:
            panel_locations = json.load(f)
        # Check if number of panels matches number of detections
        # If false return (naive approach)
        if len(panel_locations) != len(self.panel_data):
            raise Exception(
                f"Incorrect number of detections: {len(self.panel_data)} panels specified,"
                f" but {len(panel_locations)} found")

        # gather radiance values of each panel detected in the image
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        radiance_values = get_mean_radiance_values(panel_locations, img)

        reflectance_values = get_panel_factors_for_band(self.panel_data, band)

        # Assign panel to detection based on ranking of reflectance and radiance (naive)
        extraction_data = list(zip(np.sort(radiance_values), np.sort(reflectance_values)))

        # save data to file
        extraction_path, extraction_filename = get_extraction_path(image_path)
        os.makedirs((Path.cwd() / extraction_path).resolve(), exist_ok=True)
        filepath = (Path.cwd() / extraction_path / extraction_filename).resolve()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(extraction_data, f, ensure_ascii=False, indent=4)
        return filepath
