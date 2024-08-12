import os
from pathlib import Path
from typing import List, Tuple, Any

import fiona
import geopandas as gpd
import numpy as np
import rasterio
from geopandas import GeoDataFrame
from numpy import ndarray
from numpy._typing import _64Bit
from rasterio import DatasetReader
from rasterio.coords import BoundingBox
from rasterio.mask import mask
from shapely.geometry import Polygon
from tqdm import tqdm


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


def extract_using_geolocation(image, panel_location: GeoDataFrame) -> List[float]:
    # Extracts the mean intensity per band at the panel location
    panel_polygon = panel_location.unary_union.convex_hull
    out_image, out_transform = rasterio.mask.mask(image, [panel_polygon], crop=True, nodata=0)

    return [panel_band[panel_band > 0].mean() for panel_band in out_image]


def fit(intensities: ndarray[Any, np.dtype[np.floating[_64Bit] | np.float_]], expected_reflectances: List[float]) -> \
        Tuple[float, float]:
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


def save_bands(output_path, band_images, meta):
    # Combine bands back into one image
    with rasterio.open(output_path, 'w', **meta) as dst:
        for band_index, band in enumerate(band_images):
            dst.write_band(band_index + 1, band)


def get_orthophoto_paths(orthophoto_dir: str) -> List[Path]:
    template_path = Path(orthophoto_dir).resolve()
    return list(sorted([filepath for filepath in template_path.glob("*.tif")]))


def load_panel_locations(geopackage_dir) -> List[GeoDataFrame]:
    panel_locations = []
    with fiona.Env():
        for layer in fiona.listlayers(geopackage_dir):
            if layer == "android_metadata":
                continue
            panel_corner_points = gpd.read_file(geopackage_dir, layer=layer)
            panel_locations.append(panel_corner_points)
    return panel_locations


def save_orthophotos(paths: list[Path], converted_photos: list[list[ndarray]]) -> None:
    """
    This function saves the converted photos as .tif files into a new "/transformed/" directory in the images folder
    :param paths: List of orthophoto paths
    :param converted_photos: List of reflectance photos, each photo is a list of bands,
    each band is a ndarray of shape (width, height)
    """
    for path, photo in zip(paths, converted_photos):
        if photo is None:
            continue
        filename = path.as_posix().split("/")[-1].split(".")[0] + "_reflectance.tif"
        output_folder = "/".join(path.as_posix().split("/")[:-1]) + "/transformed/"
        os.makedirs(output_folder, exist_ok=True)
        output_path = output_folder + filename
        with rasterio.open(path) as original:
            meta = original.meta
        meta.update(
            dtype=rasterio.float32,
        )
        save_bands(output_path, photo, meta)


def extract_intensities_from_orthophotos(batch_of_orthophotos: list[Path],
                                         paths_with_visibility: dict[Path, ndarray],
                                         panel_locations: list[GeoDataFrame],
                                         number_of_bands: int) -> ndarray[
    Any, np.dtype[np.floating[_64Bit] | np.float_]]:
    """
    This function extracts intensities from the orthophotos.
    It does so by looking at each photo and determining which panels are visible in the photo.
    If a panel is not visible in the image it is not visible in any of the bands
    as the photos are assumed to rectified and aligned.
    Therefore, extraction is skipped for that panel with `np.Nan` values saved in the output.
    Otherwise, the function uses the recorded gps location of the panel given by `panel_locations`
    to extract the mean intensity of the panel in that band for that photo
    :type number_of_bands: int
    :return: The extracted intensities for all orthophotos, with np.Nan for values that could not be found.
    """
    intensities = np.zeros((len(batch_of_orthophotos), len(panel_locations), number_of_bands))
    for photo_index, orthophoto_path in enumerate(tqdm(batch_of_orthophotos)):
        for panel_index, panel_location in enumerate(panel_locations):
            if not paths_with_visibility[orthophoto_path][panel_index]:
                intensities[photo_index][panel_index] = np.full(number_of_bands, np.NaN)
                continue
            # extract the mean intensity for each band at that panel location
            orthophoto: DatasetReader
            with rasterio.open(orthophoto_path) as orthophoto:
                panel_intensities_per_band = extract_using_geolocation(orthophoto, panel_location)
            intensities[photo_index][panel_index] = panel_intensities_per_band
    return intensities
