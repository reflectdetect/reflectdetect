import json
import os
from pathlib import Path

import cv2
import numpy as np
import shapely.geometry as sg
from rasterio.features import rasterize

from src.extractor.BaseExtractor import BaseExtractor
from src.utils.paths import get_image_band, get_extraction_path

# Expects panel locations to be a list of YOLO_OBB bounding boxes
# in the form [id, class, x1, y1, x2, y1, x3, y3, x4, y4]
def get_mean_radiance_values(panel_locations, img):
    panel_radiance_values = []
    for detection in panel_locations:
        # ignore id and class
        detection = detection[2:]
        # convert [x1, y1, x2, y2] to [(x1, y1), (x2, y2)] and instantiate polygon
        polygon = sg.Polygon(list(zip(detection, detection[1:]))[::2])
        mask = rasterize([polygon], out_shape=img.shape)
        # mean the radiance values to get a radiance value for the detection
        mean = np.ma.array(img, mask=~(mask.astype(np.bool_))).mean()
        panel_radiance_values.append(mean)
    return panel_radiance_values


class Extractor(BaseExtractor):

    def __init__(self, panel_data):
        # Load panel data
        self.panel_data = panel_data

    def get_panel_factors_for_band(self, band):
        return [panel["bands"][band]["factor"] for panel in self.panel_data]

    def extract(self, image_path: str, detection_path: str, _=None) -> str:
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

        reflectance_values = self.get_panel_factors_for_band(band)

        # Assign panel to detection based on ranking of reflectance and radiance (naive)
        extraction_data = list(zip(np.sort(radiance_values), np.sort(reflectance_values)))

        # save data to file
        extraction_path, extraction_filename = get_extraction_path(image_path)
        os.makedirs((Path.cwd() / extraction_path).resolve(), exist_ok=True)
        filepath = (Path.cwd() / extraction_path / extraction_filename).resolve()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(extraction_data, f, ensure_ascii=False, indent=4)
        return filepath
