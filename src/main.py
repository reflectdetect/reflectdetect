import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def detect(orthophoto, panel_locations: list) -> bool:
    # use panel locations to check if a panel lies inside the orthophoto
    # return True if all panels are in the image
    pass


def extract(orthphoto, panel_locations) -> list:
    # assumes that the panels are present inside the orthophoto
    # extract mean intensity for each panel using the location
    # returns intensity for each panel
    pass


def fit(panel_intensities, panel_properties) -> list:
    # Use collected intensities and expected properites of the panels
    # to fit a function which converts the DNs to reflectance.
    #coef = np.polyfit(x, y, 1)
    pass

def interpolate(coefficients) -> dict:
    # Takes a list of linear function coefficients for each image
    # or None if the image did not contain the panels in temporal order
    # Creates missing coefficients by linearly interpolating between the coefficients
    pass

def convert(filename: str, coeffs: list):
    # converts a photo based on a linear transformation.
    pass

def load_orthophoto():
    pass


def run_pipeline_for_orthophotos(orthophotos_dir: str, panel_properties_file: str):
    template_path = (Path.cwd() / orthophotos_dir).resolve()
    with open(panel_properties_file) as f:
        panel_properties = json.load(f)
        panel_locations = [panel["location"] for panel in panel_properties]
    coefficients = {}
    for filename in template_path.glob("*"):
        photo = load_orthophoto()
        # detect which orthophotos contain panels
        if not detect(photo, panel_locations):
            continue
        # extract panel intensity for each image with panels
        panel_intensities = extract(filename, panel_locations)
        # fit linear function for each image with panels
        coeffs = fit(panel_intensities, panel_properties)
        coefficients[filename] = coeffs
    # interpolate between linear functions for images without panels
    coefficients = interpolate(coefficients)
    # convert orthophotos
    for file, coeffs in coefficients.items():
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
