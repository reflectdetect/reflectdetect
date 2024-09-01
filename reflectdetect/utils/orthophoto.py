from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import rasterio
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from rasterio import DatasetReader
from rasterio.coords import BoundingBox
from rasterio.mask import mask
from rich.progress import Progress
from shapely.geometry import Polygon

from reflectdetect.constants import ORTHOPHOTO_FOLDER
from reflectdetect.utils.debug import ProgressBar
from reflectdetect.utils.paths import get_output_path
from reflectdetect.utils.polygons import shrink_or_swell_shapely_polygon


def is_panel_in_orthophoto(orthophoto_path: Path, panel: GeoDataFrame) -> bool:
    if panel.empty:
        raise Exception("Invalid panel location, no corner points included")
    with (rasterio.open(orthophoto_path) as orthophoto):
        bounds = BoundingBox(*orthophoto.bounds)
        crs = orthophoto.crs
    orthophoto_polygon = Polygon([
        (bounds.left, bounds.bottom), (bounds.left, bounds.top),
        (bounds.right, bounds.top), (bounds.right, bounds.bottom)
    ])

    # TODO: add input for alternative crs
    # Create a GeoDataFrame for the orthophoto polygon with its CRS
    orthophoto_box = gpd.GeoDataFrame({'geometry': [orthophoto_polygon]}, crs=crs)

    # Ensure the GeoDataframe has the same CRS as the orthophoto box
    if panel.crs.name != orthophoto_box.crs.name:
        print("CRS mismatch, converting ", panel.crs, "to", orthophoto_box.crs)
        panel = panel.to_crs(orthophoto_box.crs)

    # Check if all corner points of the panel are within the orthophoto bounds
    return bool(panel.within(orthophoto_polygon).all())


def extract_using_geolocation(image: DatasetReader, panel_location: GeoDataFrame, shrink_factor: float) -> list[float]:
    # Extracts the mean intensity per band at the panel location
    panel_polygon = panel_location.union_all().convex_hull
    panel_polygon = shrink_or_swell_shapely_polygon(panel_polygon, shrink_factor)
    out_image, out_transform = rasterio.mask.mask(image, [panel_polygon], crop=True, nodata=0)

    return [panel_band[panel_band > 0].mean() for panel_band in out_image]


def save_bands(output_path: Path, band_images: list[NDArray[np.float64]], meta: dict[str, str]) -> None:
    # Combine bands back into one image
    with rasterio.open(output_path, 'w', **meta) as dst:
        for band_index, band in enumerate(band_images):
            dst.write_band(band_index + 1, band)


def get_orthophoto_paths(dataset: Path) -> list[Path]:
    return list(sorted([filepath for filepath in (dataset / ORTHOPHOTO_FOLDER).glob("*.tif")]))


def load_panel_locations(dataset: Path, geopackage_filepath: Path | None) -> list[tuple[str, GeoDataFrame]]:
    if geopackage_filepath is None:
        canonical_filename = "panel_locations.gpkg"
        path = dataset / canonical_filename
        if not path.exists():
            raise ValueError("No panel locations file found at {}.".format(path))
    else:
        path = geopackage_filepath

    panel_locations = []
    with fiona.Env():
        for layer in fiona.listlayers(path):
            if layer == "android_metadata":
                continue
            panel_corner_points = gpd.read_file(path, layer=layer)
            panel_locations.append((layer, panel_corner_points))
    return panel_locations


def save_orthophotos(paths: list[Path], converted_photos: list[list[NDArray[np.float64]] | None],
                     progress: Progress | None = None) -> None:
    """
    This function saves the converted photos as .tif files into a new "/transformed/" directory in the images folder
    :param progress:
    :param paths: list of orthophoto paths
    :param converted_photos: list of reflectance photos, each photo is a list of bands,
    each band is a ndarray of shape (width, height)
    """
    with ProgressBar(progress, "Saving orthophotos", len(converted_photos)) as pb:
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
            pb.update()


def extract_intensities_from_orthophotos(batch_of_orthophotos: list[Path],
                                         paths_with_visibility: dict[Path, NDArray[np.bool]],
                                         panel_locations: list[GeoDataFrame],
                                         number_of_bands: int,
                                         shrink_factor: float,
                                         progress: Progress | None = None) -> NDArray[np.float64]:
    """
    This function extracts intensities from the orthophotos.
    It does so by looking at each photo and determining which panels are visible in the photo.
    If a panel is not visible in the image it is not visible in any of the bands
    as the photos are assumed to rectified and aligned.
    Therefore, extraction is skipped for that panel with `np.Nan` values saved in the output.
    Otherwise, the function uses the recorded gps location of the panel given by `panel_locations`
    to extract the mean intensity of the panel in that band for that photo
    :param progress:
    :param panel_locations:
    :param paths_with_visibility:
    :param batch_of_orthophotos:
    :type number_of_bands: int
    :return: The extracted intensities for all orthophotos, with np.Nan for values that could not be found.
    """
    task = None
    if progress is not None:
        task = progress.add_task("Extracting intensities", total=len(batch_of_orthophotos), leave=False)
    intensities = np.zeros((len(batch_of_orthophotos), len(panel_locations), number_of_bands))
    for photo_index, orthophoto_path in enumerate(batch_of_orthophotos):
        for panel_index, panel_location in enumerate(panel_locations):
            if not paths_with_visibility[orthophoto_path][panel_index]:
                intensities[photo_index][panel_index] = np.full(number_of_bands, np.nan)
                continue
            # extract the mean intensity for each band at that panel location
            orthophoto: DatasetReader
            with rasterio.open(orthophoto_path) as orthophoto:
                panel_intensities_per_band = extract_using_geolocation(orthophoto, panel_location, shrink_factor)
            intensities[photo_index][panel_index] = panel_intensities_per_band
        if progress is not None and task is not None:
            progress.update(task, advance=1)
    if progress is not None and task is not None:
        progress.remove_task(task)
    return intensities
