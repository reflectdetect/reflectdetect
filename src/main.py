import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import fiona
import geopandas as gpd
import numpy as np
import rasterio
from geopandas import GeoDataFrame
from rasterio.coords import BoundingBox
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


def filter_images_with_all_panels(filepaths: List[Path], geopoints_gdfs: List[GeoDataFrame]) -> List[Path]:
    result_files: List[Path] = []

    for filepath in filepaths:
        with (rasterio.open(filepath) as orthophoto):
            # Extract the bounding box of the orthophoto
            bounds = BoundingBox(*orthophoto.bounds)
            orthophoto_polygon = Polygon([
                (bounds.left, bounds.bottom),
                (bounds.left, bounds.top),
                (bounds.right, bounds.top),
                (bounds.right, bounds.bottom)
            ])
            # TODO: add input for alternative crs
            # Create a GeoDataFrame for the orthophoto polygon with its CRS
            orthophoto_box = gpd.GeoDataFrame({
                'geometry': [orthophoto_polygon]},
                crs="EPSG:4326"
            )

            layers_found = 0
            for panel_gdf in geopoints_gdfs:
                if panel_gdf.empty:
                    continue

                # Ensure the GeoDataframe has the same CRS as the orthophoto box
                if panel_gdf.crs != orthophoto_box.crs:
                    panel_gdf = panel_gdf.to_crs(orthophoto_box.crs)
                # Check if all points in the layer are within the orthophots bounds
                all_points_within = panel_gdf.within(orthophoto_polygon).all()
                if all_points_within:
                    layers_found += 1
                    if layers_found >= 2:  # minimum for Panels for later calibration
                        result_files.append(filepath)
                        break
    # Further processing for each geopoints_gdf
    return result_files


def extract(image: Path, panel_location: GeoDataFrame) -> float:
    # image is one band of the orthophoto
    # assumes that the panel is present inside the image
    # extract mean intensity
    raise NotImplementedError


def fit(intensities: List[float], expected_reflectances: List[float]) -> Tuple[float, float]:
    slope, intersect = np.polyfit(intensities, expected_reflectances, 1)
    return slope, intersect


def interpolate(coefficients: Dict[Path, List[Tuple[float, float]]]) -> Dict[Path, List[Tuple[float, float]]]:
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


def convert(image: Path, coeffs_for_each_band: List[Tuple[float, float]]):
    # converts a photo based on a linear transformation.
    raise NotImplementedError


def load_orthophotos(orthophoto_dir: str) -> List[Path]:
    template_path = Path(orthophoto_dir).resolve()
    return [filepath for filepath in template_path.glob("*.tif")]


def load_geopackage(geopackage_dir) -> List[GeoDataFrame]:
    geopoints = []
    with fiona.Env():
        layers = fiona.listlayers(geopackage_dir)
        for layer in layers:
            if layer == "android_metadata":
                continue
            gdf = gpd.read_file(geopackage_dir, layer=layer)
            geopoints.append(gdf)
    return geopoints


def get_bands(image: Path) -> list:
    # split the orthophoto into bands
    raise NotImplementedError


def get_band_reflectance(panels, band_index) -> list:
    # return the reflectance values of each panel at a given band
    raise NotImplementedError


def run_pipeline_for_orthophotos(orthophotos_dir: str, panel_properties_file: str, geopackage_file: str):
    with open(panel_properties_file) as f:
        panels = json.load(f)

    geopoints = load_geopackage(geopackage_file)

    # Load orthophoto paths
    orthophoto_files = load_orthophotos(orthophotos_dir)

    # Detect panels in the orthophotos
    matching_files = filter_images_with_all_panels(orthophoto_files, geopoints)
    print("Found all panels in ", len(matching_files), "/", len(orthophoto_files), "files")

    # Load orthophotos with data for further processing
    coefficients: Dict[Path, List[Tuple[float, float]] | None] = {}
    for filename in orthophoto_files:
        coefficients[filename] = None
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
    parser.add_argument("images", help="Path to the image files", type=str)
    parser.add_argument("panel_properties", help="Path to the property file of the panels", type=str)
    parser.add_argument("panel_corners", help="Path to the GeoPackage file", type=str)
    args = parser.parse_args()

    run_pipeline_for_orthophotos(args.images, args.panel_properties, args.panel_corners)
