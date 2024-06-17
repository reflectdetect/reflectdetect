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
    # Creates missing coefficients (None) by linearly interpolating between the coefficients
    # `coefficients` is a dict {filename: [slope, intersect]}
    # Takes a list of linear function coefficients
    # or None in temporal order
    # TODO: Make global sorting function
    filenames = list(sorted(coefficients.keys()))  # Assuming alphabetical order makes sense
    values = [coefficients[filename] for filename in filenames]
    is_none = [v is None for v in values]  # Track which indices have value None
    non_none_vals = [(i, v) for i, v in enumerate(values) if v is not None]

    for i, _ in enumerate(values):
        if is_none[i]:  # If our value is None, interpolate
            # Find the closest indices with value on either side
            lower_idx = max(idx for idx, v in non_none_vals if idx < i)
            upper_idx = min(idx for idx, v in non_none_vals if idx > i)

            lower_coeffs = values[lower_idx]
            upper_coeffs = values[upper_idx]

            # This might be confusing as we are linearly interpolating linear functions
            # First we interpolate the slopes,
            # then the intercepts by calculating a slope and intercept for the calculation

            # Interpolate slopes
            slope = (upper_coeffs[0] - lower_coeffs[0]) / (upper_idx - lower_idx)
            intercept = lower_coeffs[0] - slope * lower_idx

            interpolated_slopes = [slope * i + intercept for i in range(lower_idx, upper_idx + 1)]

            # Interpolate intercepts
            slope = (upper_coeffs[1] - lower_coeffs[1]) / (upper_idx - lower_idx)
            intercept = lower_coeffs[1] - slope * lower_idx

            interpolated_intercepts = [slope * i + intercept for i in range(lower_idx, upper_idx + 1)]
            interpolated_values = list(zip(interpolated_slopes, interpolated_intercepts))
            # Update the values array with the interpolated values
            for j in range(lower_idx, upper_idx + 1):
                values[j] = interpolated_values[j - lower_idx]
    return dict(zip(filenames, values))


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
