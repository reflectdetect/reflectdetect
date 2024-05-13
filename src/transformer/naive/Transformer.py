import json
import os
from pathlib import Path

import cv2
import numpy as np

from src.transformer.BaseTransformer import BaseTransformer
from src.utils.paths import get_transformed_path


class Transformer(BaseTransformer):
    def transform(self, image_path, extraction_path) -> str:
        # Load extraction data
        with open(extraction_path) as f:
            extraction_data = json.load(f)

        # calculate linear model ax+b based on ref rad
        x = [panel[0] for panel in extraction_data]
        y = [panel[1] for panel in extraction_data]
        coef = np.polyfit(x, y, 1)
        poly1d_fn = np.poly1d(coef)

        # transform image
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # poly1d function converts the image to reflectance so 0.0 to 1.0
        float_img = np.vectorize(poly1d_fn)(img)

        # We want to save the image with 4 places of precision so
        factor = 10000
        transformed_img = (float_img * factor).astype(np.uint16)
        # The image will have to be multiplied by factor in order to get the reflectance value

        # save image
        transformed_img_path, transformed_img_filename = get_transformed_path(image_path)
        os.makedirs((Path.cwd() / transformed_img_path).resolve(), exist_ok=True)
        filepath = (Path.cwd() / transformed_img_path / transformed_img_filename).resolve()
        print(filepath)
        cv2.imwrite(str(filepath.resolve()), transformed_img)
        return filepath
