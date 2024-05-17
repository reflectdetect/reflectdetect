import json
import os
from pathlib import Path

import cv2
import numpy as np

from src.extractor.BaseBatchExtractor import BaseBatchExtractor
from src.extractor.shared.shared import get_mean_radiance_values
from src.utils.paths import get_image_band, get_extraction_path


class BatchExtractor(BaseBatchExtractor):

    def __init__(self, panel_data):
        # Load panel data
        self.panel_data = panel_data

    def get_panel_factors_for_band(self, band):
        return [panel["bands"][band]["factor"] for panel in self.panel_data]

    def extract(self, image_paths: [str], detection_path: str, _=None) -> str:
        # Load detection results
        with open(detection_path) as f:
            panel_locations = json.load(f)

        data = []
        for image_path in image_paths:
            # get band identifier from image path
            band = get_image_band(image_path)

            # Check if number of panels matches number of detections
            # If false return (naive approach)
            if len(panel_locations[band]) != len(self.panel_data):
                raise Exception(
                    f"Incorrect number of detections: {len(self.panel_data)} panels specified,"
                    f" but {len(panel_locations)} found")

            # gather radiance values of each panel detected in the image
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            radiance_values = get_mean_radiance_values(panel_locations, img)

            reflectance_values = self.get_panel_factors_for_band(band)

            # Assign panel to detection based on ranking of reflectance and radiance (naive)
            extraction_data = list(zip(np.sort(radiance_values), np.sort(reflectance_values)))
            data.append(extraction_data)

        # save data to file
        extraction_path, extraction_filename = get_extraction_path(image_paths[0])
        os.makedirs((Path.cwd() / extraction_path).resolve(), exist_ok=True)
        filepath = (Path.cwd() / extraction_path / extraction_filename).resolve()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return filepath
