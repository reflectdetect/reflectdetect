from pathlib import Path
from typing import List, Any

import fiona
import geopandas as gpd
import numpy as np
import rasterio
from geopandas import GeoDataFrame
from numpy import ndarray
from rasterio import DatasetReader
from rasterio.coords import BoundingBox
from rasterio.mask import mask
from shapely.geometry import Polygon
from tqdm import tqdm

from utils.paths import get_output_path


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


def save_bands(output_path, band_images, meta):
    # Combine bands back into one image
    with rasterio.open(output_path, 'w', **meta) as dst:
        for band_index, band in enumerate(band_images):
            dst.write_band(band_index + 1, band)


def get_orthophoto_paths(dataset_path: str) -> List[Path]:
    folder = "orthophotos"
    template_path = Path(dataset_path + "/" + folder).resolve()
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
        output_path = get_output_path(path, "reflectance", "transformed")
        with rasterio.open(path) as original:
            meta = original.meta
        meta.update(
            dtype=rasterio.float32,
        )
        save_bands(output_path, photo, meta)


def extract_intensities_from_orthophotos(batch_of_orthophotos: list[Path],
                                         paths_with_visibility: dict[Path, ndarray],
                                         panel_locations: list[GeoDataFrame],
                                         number_of_bands: int) -> ndarray[Any, np.dtype[np.float64]]:
    """
    This function extracts intensities from the orthophotos.
    It does so by looking at each photo and determining which panels are visible in the photo.
    If a panel is not visible in the image it is not visible in any of the bands
    as the photos are assumed to rectified and aligned.
    Therefore, extraction is skipped for that panel with `np.Nan` values saved in the output.
    Otherwise, the function uses the recorded gps location of the panel given by `panel_locations`
    to extract the mean intensity of the panel in that band for that photo
    :param panel_locations:
    :param paths_with_visibility:
    :param batch_of_orthophotos:
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
