import warnings
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.plot
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from rasterio import DatasetReader
from rasterio.coords import BoundingBox
from rasterio.mask import mask
from rich.progress import Progress
from shapely.geometry import Polygon

from reflectdetect.constants import ORTHOPHOTO_FOLDER, PANEL_LOCATIONS_FILENAME, COMPRESSION_FACTOR
from reflectdetect.utils.debug import ProgressBar
from reflectdetect.utils.iterators import get_next
from reflectdetect.utils.panel import get_panel_intensity
from reflectdetect.utils.paths import get_output_path
from reflectdetect.utils.polygons import shrink_shapely_polygon


def is_panel_in_orthophoto(orthophoto_path: Path, panel: GeoDataFrame, shrink_factor: float,
                           no_data_value: int) -> bool:
    """
    Checks if a panel is in an orthophoto based on its coordinates
    :param orthophoto_path: path to the orthophoto tiff file
    :param panel: geodataframe containing the coordinates of the panel
    :return: whether the coordinates are in the bounds of the orthophoto
    """
    if panel.empty:
        raise Exception("Invalid panel location, no corner points included")

    with rasterio.open(orthophoto_path) as orthophoto:
        bounds = BoundingBox(*orthophoto.bounds)
        orthophoto_polygon = Polygon(
            [
                (bounds.left, bounds.bottom),
                (bounds.left, bounds.top),
                (bounds.right, bounds.top),
                (bounds.right, bounds.bottom),
            ]
        )
        # Check if all corner points of the panel are within the orthophoto bounds
        if not bool(panel.within(orthophoto_polygon).all()):
            return False

        panel_polygon = panel.union_all().convex_hull
        panel_polygon = shrink_shapely_polygon(panel_polygon, shrink_factor)
        out_image, out_transform = rasterio.mask.mask(
            orthophoto, [panel_polygon], crop=True, nodata=no_data_value
        )
        # Check if any of the bands include only no data value
        if np.array([(band == no_data_value).all() for band in out_image]).any():
            return False

    return True


def extract_using_geolocation(
        photo: DatasetReader, panel_location: GeoDataFrame, shrink_factor: float, no_data_value: int
) -> list[float]:
    """
    Extract the mean intensity values from an orthophoto inside a polygon give by 4 corner points of a panel

    :param no_data_value: the value indicating that a part of the orthophoto is not data
    (typically max of the datatype or 0)
    :param photo: the orthophoto to take the intensity values from
    :param panel_location: the 4 corner points in a geodataframe
    :param shrink_factor: factor to shrink the polygon by to avoid bleed or similar artifacts
    :return: the mean intensity at that location for each band of the orthophoto
    """
    # Extracts the mean intensity per band at the panel location
    panel_polygon = panel_location.union_all().convex_hull
    panel_polygon = shrink_shapely_polygon(panel_polygon, shrink_factor)
    out_image, out_transform = rasterio.mask.mask(
        photo, [panel_polygon], crop=True, nodata=no_data_value
    )

    return [get_panel_intensity(np.ma.masked_equal(panel_band, no_data_value)) for panel_band in out_image]


def save_bands(
        output_path: Path, band_images: list[NDArray[np.float64]], meta: dict[str, str]
) -> None:
    """
    Save the bands to a new file
    :param output_path: path to the new file
    :param band_images: list of images, one for each band
    :param meta: metadata to be saved into the image
    """
    # Combine bands back into one image
    with rasterio.open(output_path, "w", **meta) as dst:
        for band_index, band in enumerate(band_images):
            band[band < 0] = 0
            with warnings.catch_warnings():
                # Ignore "RuntimeWarning: invalid value encountered in cast"
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                scaled_to_int = np.array(band * COMPRESSION_FACTOR, dtype=np.uint16)
            dst.write_band(band_index + 1, scaled_to_int)


def get_orthophoto_paths(dataset: Path, orthophotos_folder: Path | None) -> list[Path]:
    """
    Gets all the .tiff images in a folder. Uses the canonical path in the dataset if no specific path is given
    :param dataset: path to the dataset folder
    :param orthophotos_folder: name of the subfolder containing the orthophotos
    :return: list of path to the .tiff photos
    """
    if orthophotos_folder is None:
        path = dataset / ORTHOPHOTO_FOLDER
        if not path.exists():
            raise ValueError(f"No images folder found at {path}.")
    else:
        path = orthophotos_folder

    return list(sorted(list(path.glob("*.tif"))))


def load_panel_locations(
        dataset: Path, geopackage_filepath: Path | None
) -> list[tuple[str, GeoDataFrame]]:
    """
    Loads the panel location file. The default layer "android_metadata" is ignored
    Uses the canonical path in the dataset if no specific path is set
    :param dataset: path to the dataset folder containing the canonical file
    :param geopackage_filepath: specific path to the use instead
    :return: the panel locations with their respective layer name
    """
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


def build_batches_per_full_visibility(
        paths_with_visibility: dict[Path, NDArray[np.bool]],
) -> list[tuple[bool, list[Path]]]:
    """
    Batches a list of images based on a precomputed property indicating which panels are visible.
    A Batch should include a full datapoint at the start and end when possible.
    A datapoint is considered full if all panels are visible.
    :param paths_with_visibility: The paths with their visibility values
    :return: batches of paths
    """
    batches = []
    # Batch contains bool to signify whether the first path is there only for interpolation
    current_batch: tuple[bool, list[Path]] = (False, [])
    for (path, panel_visibility), nextVisibility in get_next(
            paths_with_visibility.items()
    ):
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


def save_orthophotos(
        dataset: Path,
        paths: list[Path],
        converted_photos: list[list[NDArray[np.float64]] | None],
        progress: Progress | None = None,
) -> None:
    """
    This function saves the converted photos as .tif files into a new "/transformed/" directory in the images folder
    :param progress: a optional progress bar to be updated
    :param paths: list of orthophoto paths
    :param converted_photos: list of reflectance photos, each photo is a list of bands,
    each band is a ndarray of shape (width, height)
    """
    with ProgressBar(progress, "Saving orthophotos", len(converted_photos)) as pb:
        for path, photo in zip(paths, converted_photos):
            if photo is None:
                continue
            output_path = get_output_path(
                dataset, path, "reflectance.tif", "transformed"
            )
            with rasterio.open(path) as original:
                meta = original.meta
            meta.update(
                dtype=rasterio.float32,
            )
            save_bands(output_path, photo, meta)
            pb.update()
