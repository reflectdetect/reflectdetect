import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from geopandas import GeoDataFrame
from numpy import ndarray
from rasterio.coords import BoundingBox
from rasterio.windows import from_bounds
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


def is_panel_in_orthophoto(orthophoto_path: Path, panel: GeoDataFrame) -> bool:
    with (rasterio.open(orthophoto_path) as orthophoto):
        bounds = BoundingBox(*orthophoto.bounds)
        orthophoto_polygon = Polygon([
            (bounds.left, bounds.bottom), (bounds.left, bounds.top),
            (bounds.right, bounds.top), (bounds.right, bounds.bottom)
        ])

        # TODO: add input for alternative crs
        # Create a GeoDataFrame for the orthophoto polygon with its CRS
        orthophoto_box = gpd.GeoDataFrame({'geometry': [orthophoto_polygon]}, crs="EPSG:4326")

        if panel.empty:
            print("Invalid panel location, no corner points included")
            return False

        # Ensure the GeoDataframe has the same CRS as the orthophoto box
        if panel.crs != orthophoto_box.crs:
            print("CRS mismatch, converting ", panel.crs, "to", orthophoto_box.crs)
            panel = panel.to_crs(orthophoto_box.crs)

        # Check if all corner points of the panel are within the orthophoto bounds
        return panel.within(orthophoto_polygon).all()


def extract(band_image, panel_location: GeoDataFrame) -> float:
    # TODO: Check if bounding are correct.
    panel_bounds = panel_location.geometry.bounds
    window = from_bounds(*panel_bounds, band_image.transform)
    panel_data = band_image[window.toslices()]
    return panel_data.mean()


def fit(intensities: List[float], expected_reflectances: List[float]) -> Tuple[float, float]:
    slope, intersect = np.polyfit(intensities, expected_reflectances, 1)
    return slope, intersect


def interpolate(values: ndarray[float]) -> ndarray[float]:
    is_none = [np.isnan(v) for v in values]
    non_none_vals = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]

    if len(non_none_vals) <= 0:
        print('No values found for interpolation.')
        return values

    for i, _ in enumerate(values):
        if is_none[i]:  # If our value is None, interpolate
            # Find the closest indices with value on either side
            lower_idx = max(idx for idx, v in non_none_vals if idx < i)
            upper_idx = min(idx for idx, v in non_none_vals if idx > i)

            lower_value = values[lower_idx]
            upper_value = values[upper_idx]

            slope = (upper_value - lower_value) / (upper_idx - lower_idx)
            intercept = lower_value - slope * lower_idx

            interpolated_values = [slope * i + intercept for i in range(lower_idx, upper_idx + 1)]

            # Update the values array with the interpolated values
            for j in range(lower_idx, upper_idx + 1):
                values[j] = interpolated_values[j - lower_idx]
    return values


def convert(image: Path, coeffs_for_each_band: List[Tuple[float, float]]):
    # converts a photo based on a linear transformation.
    raise NotImplementedError


def get_orthophoto_paths(orthophoto_dir: str) -> List[Path]:
    template_path = Path(orthophoto_dir).resolve()
    return [filepath for filepath in template_path.glob("*.tif")]


def load_panel_locations(geopackage_dir) -> List[GeoDataFrame]:
    panel_locations = []
    with fiona.Env():
        for layer in fiona.listlayers(geopackage_dir):
            if layer == "android_metadata":
                continue
            panel_corner_points = gpd.read_file(geopackage_dir, layer=layer)
            panel_locations.append(panel_corner_points)
    return panel_locations


def get_bands(image: Path) -> list:
    # split the orthophoto into bands
    raise NotImplementedError


def get_band_reflectance(panels, band_index) -> list:
    # return the reflectance values of each panel at a given band
    raise NotImplementedError


def load_panel_properties(panel_properties_file):
    with open(panel_properties_file) as f:
        panels = json.load(f)
    return panels


def run_pipeline_for_orthophotos(orthophotos_dir: str, panel_properties_file: str, geopackage_file: str):
    # Load necessary data
    panels = load_panel_properties(panel_properties_file)
    panel_locations = load_panel_locations(geopackage_file)
    orthophoto_paths = get_orthophoto_paths(orthophotos_dir)
    number_of_bands = len(get_bands(orthophoto_paths[0]))
    assert len(panels) == len(panel_locations)

    # 1. For each image extract each panel intensities for each band

    intensities = pd.DataFrame(columns=['orthophoto_path', 'panel_index', 'band', 'intensity'])

    for orthophoto_path in orthophoto_paths:
        intensities[orthophoto_path] = np.full((len(panels), number_of_bands), None)
        for panel_index, panel in enumerate(panel_locations):
            if not is_panel_in_orthophoto(orthophoto_path, panel):
                continue
            if photo is None:
                with rasterio.open(orthophoto_path) as orthophoto:
                    photo = orthophoto.read()
            for band, band_index in get_bands(photo):
                panel_intensity = extract(band, panel)
                intensities[orthophoto_path][panel_index][band_index] = panel_intensity

    intensities = interpolate(intensities)
    for file, intensities_per_band in intensities.items():
        coeffs = fit(intensities_per_band, get_band_reflectance(panels, ))
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
