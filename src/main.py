import argparse
import json
import logging
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.coords import BoundingBox
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


def filter_images_with_all_panels(filepaths, geopoints_gdfs):
    result_files = []
    for panel_gdf in geopoints_gdfs:
        for filepath in filepaths:
            with rasterio.open(filepath) as orthophoto:
                # TODO: robust check wether panels are in image
                bounds = BoundingBox(*orthophoto.bounds)
                orthophoto_box = gpd.GeoDataFrame({'geometry': [Polygon(
                    [(bounds.left, bounds.bottom), (bounds.left, bounds.top), (bounds.right, bounds.top),
                     (bounds.right, bounds.bottom)])]})

                # TODO: also get panel data from images with only some panels visible
                all_points_within = True
                for _, point in panel_gdf.iterrows():
                    if 'geometry' in point and point.geometry is not None:
                        if not orthophoto_box.contains(point.geometry).all():
                            all_points_within = False
                            break
                if all_points_within:
                    result_files.append(filepath)
                # Further processing for each geopoints_gdf
    return result_files


def extract(image, panel_location) -> float:
    # image is one band of the orthophoto
    # assumes that the panel is present inside the image
    # extract mean intensity
    raise NotImplementedError


def fit(intensities, expected_reflectances) -> list:
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
    raise NotImplementedError


def load_orthophotos(orthophoto_dir):
    template_path = Path(orthophoto_dir).resolve()
    return [filepath for filepath in template_path.glob("*.tif")]


def load_geopackage(geopackage_dir):
    geopoints = []
    with fiona.Env():
        layers = fiona.listlayers(geopackage_dir)
        for layer in layers:
            gdf = gpd.read_file(geopackage_dir, layer=layer)
            geopoints.append(gdf)
    return geopoints


def get_bands(orthophoto) -> list:
    # split the orthophoto into bands
    raise NotImplementedError


def get_band_reflectance(panels, band_index) -> list:
    # return the reflectance values of each panel at a given band
    raise NotImplementedError


def run_pipeline_for_orthophotos(orthophotos_dir: str, panel_properties_file: str, geopackage_file: str, shape: str,
                                 side_length_meters: float):
    with open(panel_properties_file) as f:
        panels = json.load(f)

    geopoints = load_geopackage(geopackage_file)
    geopoints_gdf = gpd.GeoDataFrame(pd.concat(geopoints, ignore_index=True))
    geopoints_gdf.to_file('geopoints.geojson', driver='GeoJSON')

    # Load orthophoto paths
    orthophoto_files = load_orthophotos(orthophotos_dir)

    # Detect panels in the orthophotos
    matching_files = filter_images_with_all_panels(orthophoto_files, geopoints_gdf)
    print("Found all panels in ", len(matching_files), "/", len(orthophoto_files), "files")

    # Load orthophotos with data for further processing
    coefficients = {}
    for filename in matching_files:
        with rasterio.open(filename) as dataset:
            photo = dataset.read()
            coefficients[filename] = []
            for band, band_index in get_bands(photo):
                panel_intensities = [extract(band, panel["location"]) for panel in panels]
                coefficients[filename].append(fit(panel_intensities, get_band_reflectance(panels, band_index)))

    coefficients = interpolate(coefficients)
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
    parser.add_argument("geopackage_file", help="Path to the GeoPackage file", type=str)
    parser.add_argument("shape", help="Shape of the panels (sq, rect, circ)", type=str)
    parser.add_argument("side_length_meters", help="Side length of the panels in meters", type=float)
    args = parser.parse_args()

    run_pipeline_for_orthophotos(args.path, args.panel_properties, args.geopackage_file, args.shape,
                                 args.side_length_meters)
