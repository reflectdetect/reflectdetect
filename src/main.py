import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def detect(orthophoto, panel_location) -> bool:
    # use panel locations to check if a panel lies inside the orthophoto
    pass


def extract(image, panel_location) -> float:
    # image is one band of the orthophoto
    # assumes that the panel is present inside the image
    # extract mean intensity
    pass


def fit(intensities, expected_reflectances) -> list:
    # Use collected intensities and expected properties of the panels
    # to fit a function which converts the DNs to reflectance.
    coeffs = np.polyfit(intensities, expected_reflectances, 1)
    return coeffs.tolist()


def interpolate(coefficients) -> dict:
    # Takes a list of linear function coefficients
    # or None in temporal order
    # Creates missing coefficients (None) by linearly interpolating between the coefficients
    pass


def convert(orthophoto, coeffs_for_each_band):
    # converts a photo based on a linear transformation.
    pass


def load_orthophoto(filename):
    pass


def get_bands(orthophoto) -> list:
    # split the orthophoto into bands
    pass


def get_band_reflectance(panels, band_index) -> list:
    # return the reflectance values of each panel at a given band
    pass


def get_coefficients_for_orthophoto(filename, panels):
    photo = load_orthophoto(filename)
    # detect which orthophotos contain all panels
    if not np.array([detect(photo, panel["location"]) for panel in panels]).any():
        return None
    

    # Calculate the coefficients for each band
    coefficients = []
    for band, band_index in get_bands(photo):
        # extract panel intensity for each image with panels
        panel_intensities = [extract(band, panel["location"]) for panel in panels]
        coefficients[band_index] = fit(panel_intensities, get_band_reflectance(panels, band_index))
        # fit linear function for each image with panels
    return coefficients


def run_pipeline_for_orthophotos(orthophotos_dir: str, panel_properties_file: str):
    template_path = (Path.cwd() / orthophotos_dir).resolve()
    with open(panel_properties_file) as f:
        panels = json.load(f)

    sparse_coefficients = {}
    for filename in template_path.glob("*"):
        sparse_coefficients[filename.name] = get_coefficients_for_orthophoto(filename, panels)
    # interpolate between linear functions for images without panels
    dense_coefficients = interpolate(sparse_coefficients)
    # convert orthophotos
    for file, coeffs in dense_coefficients.items():
        convert(file, coeffs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog='ReflectDetect',
        description='Automatically detect reflection calibration panels in images and transform the given images to '
                    'reflectance',
        epilog='If you have any questions, please contact')
    parser.add_argument("path", help="Path to the image files", type=str)
    parser.add_argument("panel_properties", help="Path to the property file of the panels", type=str)
    parser.add_argument("--no-georef", help="", type=bool)
    args = parser.parse_args()

    run_pipeline_for_orthophotos(args.path, args.panel_properties)
