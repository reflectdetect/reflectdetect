import json
import os
from pathlib import Path

import cv2
import numpy as np
import shapely.geometry as sg
from rasterio.features import rasterize

from src.extractor.BaseExtractor import BaseExtractor
from src.utils.paths import get_image_band, get_extraction_path


class Extractor(BaseExtractor):
    def extract(self, image_path: str, detection_path: str, panel_data_path: str) -> str:
        # get band identifier from image path
        band = get_image_band(image_path)
        # Load panel data
        with open(panel_data_path) as f:
            panel_data = json.load(f)

        # Load detection results
        with open(detection_path) as f:
            detection_data = json.load(f)
        # Check if number of panels matches number of detections
        # If false return (naive approach)
        if len(detection_data) != len(panel_data):
            raise Exception(
                f"Incorrect number of detections: {len(panel_data)} panels specified but {len(detection_data)} detection found")

        # gather radiance values of each detection rect from image
        # load image

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        detection_radiance = []
        for detection in detection_data:
            detection = detection[2:]
            polygon = sg.Polygon(list(zip(detection, detection[1:]))[::2])
            mask = rasterize([polygon], out_shape=img.shape)
            # mean the radiance values to get a radiance value for each detection
            mean = np.ma.array(img, mask=~(mask.astype(np.bool_))).mean()
            detection_radiance.append(mean)

        panels = [panel["bands"][band]["factor"] for panel in panel_data]

        # Assign panel to detection based on ranking of reflectance and radiance (naive)
        extraction_data = list(zip(np.sort(detection_radiance), np.sort(panels)))

        # save data to file
        extraction_path, extraction_filename = get_extraction_path(image_path)
        os.makedirs((Path.cwd() / extraction_path).resolve(), exist_ok=True)
        filepath = (Path.cwd() / extraction_path / extraction_filename).resolve()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(extraction_data, f, ensure_ascii=False, indent=4)
        return filepath
