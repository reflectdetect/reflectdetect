import argparse
import fiona
import json
import geojson
import pandas as pd
import rasterio
import logging
from pathlib import Path

import numpy as np
import geopandas as gpd
from rasterio.coords import BoundingBox
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)


def detect(orthophoto_dir, geopoints_gdf, shape, side_length_meters=None):
    template_path = (Path.cwd() / orthophoto_dir).resolve()
    result_files = []

    if shape not in ['sq', 'rect', 'circ']:
        raise ValueError("shape must be either 'sq', 'rect', or 'circ'")

    # Haversine formula to calculate distance between two points in meters
    def haversine(lat1, lon1, lat2, lon2):
        R = 6378100  # Radius of the Earth in meters
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def extract_panel_location_from_image(result_files, shape, side_length_meters):
        panel_locations = []

        for filepath in result_files:
            with rasterio.open(filepath) as dataset:
                # TODO: Check if its a display problem with matplotlib otherwise add distortion correction
                #  (See test.ipynb)
                if shape == 'sq' and side_length_meters is not None:
                    pixel_size = dataset.res[0]  # Assuming square pixels
                    side_length_pixels = side_length_meters / pixel_size

                    used_points = set()

                    for i, point in geopoints_gdf.iterrows():
                        if point.geometry is not None and i not in used_points:
                            lat1, lon1 = point.geometry.y, point.geometry.x

                            for j, point2 in geopoints_gdf.iterrows():
                                if j > i and point2.geometry is not None and j not in used_points:
                                    lat2, lon2 = point2.geometry.y, point2.geometry.x

                                    # Calculate the distance between the diagonal points
                                    diagonal_distance = np.sqrt(2 * (side_length_meters ** 2))

                                    if np.isclose(haversine(lat1, lon1, lat2, lon2), diagonal_distance, atol=1.0):
                                        # Calculate the other two points
                                        dx = (lon2 - lon1) / 2
                                        dy = (lat2 - lat1) / 2

                                        # Center of the square
                                        center_lon = (lon1 + lon2) / 2
                                        center_lat = (lat1 + lat2) / 2

                                        # Calculate the other two corners
                                        point_b = Point(center_lon - dy, center_lat + dx)
                                        point_d = Point(center_lon + dy, center_lat - dx)

                                        # Convert geographic coordinates to pixel coordinates
                                        row_a, col_a = dataset.index(lon1, lat1)
                                        row_c, col_c = dataset.index(lon2, lat2)
                                        row_b, col_b = dataset.index(point_b.x, point_b.y)
                                        row_d, col_d = dataset.index(point_d.x, point_d.y)

                                        # Check if all points are within the image bounds
                                        if all(0 <= coord < dataset.width for coord in
                                               [col_a, col_b, col_c, col_d]) and all(
                                                0 <= coord < dataset.height for coord in [row_a, row_b, row_c, row_d]):
                                            panel_locations.append({
                                                "file": str(filepath),
                                                "points": [
                                                    (lon1, lat1),
                                                    (point_b.x, point_b.y),
                                                    (lon2, lat2),
                                                    (point_d.x, point_d.y)
                                                ]
                                            })
                                            used_points.update([i, j])
                                            break
                elif shape == 'rect':
                    # TODO: Implement rectangle detection
                    pass
                elif shape == 'circ':
                    # TODO: Implement circle detection
                    pass

        with open('panel_locations.json', 'w') as f:
            json.dump(panel_locations, f, indent=4)

    # Load each orthophoto
    for filepath in template_path.glob("*.tif"):
        with rasterio.open(filepath) as dataset:
            # Get the bounds of the orthophoto
            bounds = BoundingBox(*dataset.bounds)
            # Create a bounding box geometry
            orthophoto_box = gpd.GeoDataFrame(
                {'geometry': [Polygon(
                    [
                        (bounds.left, bounds.bottom),
                        (bounds.left, bounds.top),
                        (bounds.right, bounds.top),
                        (bounds.right, bounds.bottom)])
                ]
                }
            )
            # Check if all points from geopoints_gdf are within the bounds
            all_points_within = True
            for _, point in geopoints_gdf.iterrows():
                if 'geometry' in point and point.geometry is not None:
                    if not orthophoto_box.contains(point.geometry).all():
                        all_points_within = False
                        break
            if all_points_within:
                result_files.append(filepath)

    extract_panel_location_from_image(result_files, shape, side_length_meters)

    return result_files


def extract(image, panel_location) -> float:
    # image is one band of the orthophoto
    # assumes that the panel is present inside the image
    # extract mean intensity
    pass


def fit(intensities, expected_reflectances) -> list:
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


def load_orthophoto(orthophoto_dir):
    template_path = (Path.cwd() / orthophoto_dir).resolve()
    for filepath in template_path.glob("*"):
        yield filepath, gpd.read_file(filepath)


def load_geopackage(filepath):
    geopoints = []
    with fiona.Env():
        layers = fiona.listlayers(filepath)
        for layer in layers:
            gdf = gpd.read_file(filepath, layer=layer)
            geopoints.append(gdf)
    return geopoints


def get_bands(orthophoto) -> list:
    # split the orthophoto into bands
    pass


def get_band_reflectance(panels, band_index) -> list:
    # return the reflectance values of each panel at a given band
    pass


def run_pipeline_for_orthophotos(orthophotos_dir: str, panel_properties_file: str, geopackage_file: str, shape: str,
                                 side_length_meters: float):
    with open(panel_properties_file) as f:
        panels = json.load(f)

    geopoints = load_geopackage(geopackage_file)
    geopoints_gdf = gpd.GeoDataFrame(pd.concat(geopoints, ignore_index=True))
    geopoints_gdf.to_file('geopoints.geojson', driver='GeoJSON')

    coefficients = {}
    for filename, photo in load_orthophoto(orthophotos_dir):
        result_files = detect(filename, geopoints_gdf, shape, side_length_meters)
        if not result_files:
            continue

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
