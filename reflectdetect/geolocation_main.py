import json
import logging
import warnings
from pathlib import Path

import numpy as np
import rasterio
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from rasterio import DatasetReader, logging as rasterio_logging
from rich.progress import Progress, TextColumn, TimeElapsedColumn, BarColumn, MofNCompleteColumn, \
    SpinnerColumn
from rich.table import Column
from tap import Tap

from reflectdetect.PanelProperties import GeolocationPanelProperties
from reflectdetect.constants import PANEL_PROPERTIES_FILENAME
from reflectdetect.pipeline import interpolate_intensities, fit, convert
from reflectdetect.utils.debug import debug_combine_and_plot_intensities, debug_save_intensities, ProgressBar, \
    debug_show_geolocation
from reflectdetect.utils.iterators import get_next
from reflectdetect.utils.orthophoto import load_panel_locations, get_orthophoto_paths, is_panel_in_orthophoto, \
    extract_intensities_from_orthophotos, save_orthophotos
from reflectdetect.utils.panel import get_band_reflectance
from reflectdetect.utils.paths import get_output_path
from reflectdetect.utils.thread import run_in_thread

logger = logging.getLogger(__name__)

rasterio_logging.disable()


def load_panel_properties(dataset: Path, panel_properties_file: Path | None) -> list[GeolocationPanelProperties]:
    if panel_properties_file is None:
        path = dataset / PANEL_PROPERTIES_FILENAME
    else:
        path = panel_properties_file

    with open(path) as f:
        panels = [GeolocationPanelProperties.parse_obj(item) for item in json.load(f)]
    return panels


def convert_orthophotos_to_reflectance(paths: list[Path],
                                       intensities: NDArray[np.float64], progress: Progress | None = None) -> list[
    list[NDArray[np.float64]] | None]:
    """
    This function converts the intensity values to reflectance values.
    For each photo we convert each band separately
    by collecting all the intensities of the panels for the given photo and band.
    The intensities are then combined with the known reflectance values of the panels
    at the given band to fit a linear function (Empirical Line Method).
    Read more about ELM: https://www.asprs.org/wp-content/uploads/2015/05/3E%5B5%5D-paper.pdf
    :param paths: list of orthophoto paths
    :param intensities: intensity values matrix of shape (photo, panel, band)
    :return: list of reflectance photos, each photo is a list of bands, each band is a ndarray of shape (width, height)
    """
    unconverted_photos = []
    converted_photos: list[list[NDArray[np.float64]] | None] = []
    with ProgressBar(progress, "Converting photos", len(paths)) as pb:
        for photo_index, orthophoto_path in enumerate(paths):
            converted_bands = []
            orthophoto: DatasetReader
            with rasterio.open(orthophoto_path) as orthophoto:
                photo = orthophoto.read()
            for band_index, band in enumerate(photo):
                intensities_of_panels = intensities[photo_index, :, band_index]
                if np.isnan(intensities_of_panels).any():
                    # If for some reason not all intensities are present, we save the indices for debugging purposes
                    # A None value is appended to the converted photos to not lose
                    # the connection between orthophotos and converted photos based on the index
                    unconverted_photos.append((photo_index, band_index))
                    converted_photos.append(None)
                    break
                coefficients = fit(intensities_of_panels, get_band_reflectance(panel_properties, band_index))
                converted_bands.append(convert(band, coefficients))
            else:
                # For loop did not break, therefore all bands were converted
                converted_photos.append(converted_bands)
            pb.update()
        if len(unconverted_photos) > 0:
            print("WARNING: Could not convert", len(unconverted_photos), "photos")
        return converted_photos


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


def orthophoto_main(dataset: Path, panel_locations_file: Path | None, debug: bool = False) -> None:
    with Progress(SpinnerColumn(),
                  TextColumn("[progress.description]{task.description}", table_column=Column(width=40)),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), BarColumn(), MofNCompleteColumn(),
                  TextColumn("•"), TimeElapsedColumn(),
                  ) as progress:
        with warnings.catch_warnings():
            # Ignore BadApplicationID Warning (action: "once" does not seem to work)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            panel_locations = load_panel_locations(dataset, panel_locations_file)
        orthophoto_paths = get_orthophoto_paths(dataset)
        all_images_task = progress.add_task("Total Progress", total=len(orthophoto_paths))

        photo: DatasetReader
        with rasterio.open(orthophoto_paths[0]) as photo:
            number_of_bands = len(photo.read())

        # --- Input validation
        if len(panel_properties) != len(panel_locations):
            raise Exception("Number of panel specifications does not match number of panel locations")

        panel_properties_layer_names = [panel.layer_name for panel in panel_properties]
        panel_location_layer_names = [location[0] for location in panel_locations]
        if set(panel_properties_layer_names) != set(panel_location_layer_names):
            raise Exception("Panel properties layer names do not match panel locations layer names")
        if number_of_bands != len(panel_properties[0].bands):
            raise Exception(
                "Number of bands in the images does not match number of bands in the panel specification")

        # Reorder panel_locations to match panel_properties for easier indexing
        panel_order = [panel_properties_layer_names.index(layer) for layer in panel_location_layer_names]
        ordered_panel_locations: list[GeoDataFrame] = [panel_locations[i][1] for i in panel_order]

        paths_with_visibility = {}
        number_of_paths_with_visibility = 0
        with ProgressBar(progress, "Detecting visible panels", len(orthophoto_paths)) as pb:
            for path in orthophoto_paths:
                panels_visible = np.array(
                    [is_panel_in_orthophoto(path, location) for location in ordered_panel_locations]
                )
                number_of_paths_with_visibility += panels_visible.sum() > 0
                paths_with_visibility[path] = panels_visible
                pb.update()

        progress.console.print("Number of photos with panels visible: ",
                               number_of_paths_with_visibility, "/", len(orthophoto_paths)) if debug else None

        batches = build_batches_per_full_visibility(paths_with_visibility)

        output_folder = dataset / "debug"
        if debug:
            for p in (output_folder / "intensity").glob("*.csv"):
                p.unlink()
            for p in (output_folder / "panels").glob("*.tif"):
                p.unlink()
            for path, visibility in paths_with_visibility.items():
                if visibility.sum() == 0:
                    continue
                output_path = get_output_path(dataset, path, "panels.tif", "debug/panels")
                run_in_thread(debug_show_geolocation, False, path, ordered_panel_locations, visibility,
                              args.shrink_factor,
                              output_path)

        for (first_path_is_duplicate, batch) in batches:
            # --- Run pipeline
            i = extract_intensities_from_orthophotos(batch, paths_with_visibility, ordered_panel_locations,
                                                     number_of_bands,
                                                     args.shrink_factor, progress)
            if debug:
                debug_save_intensities(first_path_is_duplicate, i, number_of_bands, output_folder / "intensity")
            i = interpolate_intensities(i, number_of_bands, len(panel_properties), progress)
            if debug:
                debug_save_intensities(first_path_is_duplicate, i, number_of_bands, output_folder / "intensity",
                                       "_interpolated")
            c = convert_orthophotos_to_reflectance(batch, i, progress)
            del i
            save_orthophotos(dataset, batch, c, progress)
            del c
            progress.update(all_images_task, advance=len(batch))
    if debug:
        debug_combine_and_plot_intensities(len(orthophoto_paths), number_of_bands, len(panel_properties),
                                           output_folder)
        debug_combine_and_plot_intensities(len(orthophoto_paths), number_of_bands, len(panel_properties),
                                           output_folder,
                                           "_interpolated")


def main():
    # --- Get input arguments from user
    logging.basicConfig(level=logging.INFO)


    class GeolocationArgumentParser(Tap):
        dataset: str  # Path to the dataset folder
        panel_locations_file: str | None = None  # Path to file instead "geolocations.gpk" in the dataset folder
        panel_properties_file: str | None = None  # Path to file instead "panel_properties.json" in the dataset folder
        debug: bool = False  # Prints logs and adds debug images into a /debug/ directory in the dataset folder
        shrink_factor: float = 0.2  # How many percent to shrink the detected panel area, to avoid artifacts like bleed

        def configure(self) -> None:
            self.add_argument('dataset')
            self.add_argument('-d', '--debug')


    args = GeolocationArgumentParser(
        description='Automatically detect reflection calibration panels in images and transform the given images to '
                    'reflectance',
        epilog='If you have any questions, please contact').parse_args()

    # Input Validation
    if not Path(args.dataset).exists():
        raise Exception(f"Could not find specified dataset folder: {args.panel_locations_file}")

    if args.panel_locations_file is not None and not Path(args.panel_locations_file).exists():
        raise Exception(f"Could not find specified panel location file: {args.panel_locations_file}")

    if args.panel_properties_file is not None and not Path(args.panel_properties_file).exists():
        raise Exception(f"Could not find specified panel properties file: {args.panel_properties_file}")

    panel_properties_file = Path(args.panel_properties_file) if args.panel_properties_file is not None else None
    panel_locations_file = Path(args.panel_locations_file) if args.panel_locations_file is not None else None
    # TODO: validate panel_location_file
    dataset = Path(args.dataset) if args.dataset is not None else None
    panel_properties = load_panel_properties(dataset, panel_properties_file)
    orthophoto_main(dataset, panel_locations_file, args.debug)
