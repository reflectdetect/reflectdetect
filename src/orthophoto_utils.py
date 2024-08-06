import json
from pathlib import Path
from typing import List, Tuple, Any

import fiona
import geopandas as gpd
import numpy as np
import rasterio
from geopandas import GeoDataFrame
from numpy import ndarray
from numpy._typing import _64Bit
from rasterio.coords import BoundingBox
from rasterio.mask import mask
from shapely.geometry import Polygon


def is_panel_in_orthophoto(orthophoto_path: Path, panel: GeoDataFrame, crs: str = "EPSG:4326") -> bool:
    with (rasterio.open(orthophoto_path) as orthophoto):
        bounds = BoundingBox(*orthophoto.bounds)
        orthophoto_polygon = Polygon([
            (bounds.left, bounds.bottom), (bounds.left, bounds.top),
            (bounds.right, bounds.top), (bounds.right, bounds.bottom)
        ])

        # TODO: add input for alternative crs
        # Create a GeoDataFrame for the orthophoto polygon with its CRS
        orthophoto_box = gpd.GeoDataFrame({'geometry': [orthophoto_polygon]}, crs=crs)

        if panel.empty:
            print("Invalid panel location, no corner points included")
            return False

        # Ensure the GeoDataframe has the same CRS as the orthophoto box
        if panel.crs != orthophoto_box.crs:
            print("CRS mismatch, converting ", panel.crs, "to", orthophoto_box.crs)
            panel = panel.to_crs(orthophoto_box.crs)

        # Check if all corner points of the panel are within the orthophoto bounds
        return panel.within(orthophoto_polygon).all()


def extract(image, panel_location: GeoDataFrame) -> List[float]:
    # Extracts the mean intensity per band at the panel location
    panel_polygon = panel_location.unary_union.convex_hull
    out_image, out_transform = rasterio.mask.mask(image, [panel_polygon], crop=True, nodata=0)

    return [panel_band[panel_band > 0].mean() for panel_band in out_image]


def fit(intensities: ndarray[Any, np.dtype[np.floating[_64Bit] | np.float_]], expected_reflectances: List[float]) -> Tuple[float, float]:
    slope, intersect = np.polyfit(intensities, expected_reflectances, 1)
    return slope, intersect


def interpolate(values: ndarray) -> ndarray:
    is_none = [np.isnan(v) for v in values]
    non_none_vals = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]

    if len(non_none_vals) < 1:
        print('No values found for interpolation.')
        return values

    for i, _ in enumerate(values):
        if is_none[i]:  # If our value is None, interpolate
            # Find the closest indices with value on either side
            lower = list(idx for idx, v in non_none_vals if idx < i)
            upper = list(idx for idx, v in non_none_vals if idx > i)
            if len(lower) == 0 and len(upper) == 0:
                continue

            if len(lower) == 0:
                upper_idx = min(upper)
                upper_value = values[upper_idx]
                values[i] = upper_value
            elif len(upper) == 0:
                lower_idx = max(lower)
                lower_value = values[lower_idx]
                values[i] = lower_value
            else:
                lower_idx = max(lower)
                upper_idx = min(upper)

                lower_value = values[lower_idx]
                upper_value = values[upper_idx]

                slope = (upper_value - lower_value) / (upper_idx - lower_idx)
                intercept = lower_value - slope * lower_idx

                interpolated_values = [slope * i + intercept for i in range(lower_idx, upper_idx + 1)]

                # Update the values array with the interpolated values
                for j in range(lower_idx, upper_idx + 1):
                    values[j] = interpolated_values[j - lower_idx]
    return values


def convert(band_image: ndarray, coeffs: Tuple[float, float]) -> ndarray:
    # converts a photo based on a linear transformation.
    return np.poly1d(coeffs)(band_image)


def save(output_path, band_images, meta):
    # Combine bands back into one image
    with rasterio.open(output_path, 'w', **meta) as dst:
        for band_index, band in enumerate(band_images):
            dst.write_band(band_index + 1, band)


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
    return [reflectances['bands'][band_index]['factor'] for reflectances in panels]


def load_panel_properties(panel_properties_file):
    with open(panel_properties_file) as f:
        panels = json.load(f)
    return panels
