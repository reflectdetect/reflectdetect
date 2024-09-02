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

from reflectdetect.constants import ORTHOPHOTO_FOLDER, PANEL_LOCATIONS_FILENAME
from reflectdetect.utils.debug import ProgressBar
from reflectdetect.utils.iterators import get_next
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
        path = dataset / PANEL_LOCATIONS_FILENAME
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


def build_batches_per_full_visibility(paths_with_visibility: dict[Path, NDArray[np.bool]]) -> list[
    tuple[bool, list[Path]]]:
    batches = []
    # Batch contains bool to signify whether the first path is there only for interpolation
    current_batch: tuple[bool, list[Path]] = (False, [])
    for (path, panel_visibility), nextVisibility in get_next(paths_with_visibility.items()):
        current_batch[1].append(path)

        all_panels_visible = panel_visibility.all()
        if all_panels_visible:
            batches.append(current_batch)
            if nextVisibility is not None and nextVisibility[1].all():
                # The next image has all panels visible, so no need to include this one twice
                current_batch = (False, [])
            else:
                # This results in photos at the end of a batch being also in the next batch.
                # Therefore, they are calculated twice, but otherwise the interpolation would lose their information.
                # The next batch could start with an images without panels in it,
                # which would need the image with panels before it to be interpolated correctly
                current_batch = (True, [path])
    batches.append(current_batch)
    return batches


def save_orthophotos(dataset: Path, paths: list[Path], converted_photos: list[list[NDArray[np.float64]] | None],
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
            output_path = get_output_path(dataset, path, "reflectance.tif", "transformed")
            with rasterio.open(path) as original:
                meta = original.meta
            meta.update(
                dtype=rasterio.float32,
            )
            save_bands(output_path, photo, meta)
            pb.update()



