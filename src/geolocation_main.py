import json
import logging
import os.path
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import rasterio
from numpy import dtype, ndarray
from rasterio import DatasetReader
from tap import Tap
from tqdm import tqdm

from pipeline import fit, convert, interpolate_intensities
from utils.debug import debug_combine_and_plot_intensities, debug_save_intensities
from utils.iterators import get_next
from utils.orthophoto import load_panel_locations, get_orthophoto_paths, is_panel_in_orthophoto, \
    extract_intensities_from_orthophotos, save_orthophotos
from utils.panel import get_band_reflectance

logger = logging.getLogger(__name__)


def load_panel_properties(dataset_path: str, panel_properties_filepath: str | None):
    if panel_properties_filepath is None:
        canonical_filename = "panel_properties.json"
        path = dataset_path + "/" + canonical_filename
        if not os.path.exists(path):
            raise ValueError("No panel properties file found at {}.".format(path))
    else:
        path = panel_properties_filepath

    with open(path) as f:
        panels = json.load(f)
        # TODO verify
    return panels


def convert_orthophotos_to_reflectance(paths: List[Path],
                                       intensities: ndarray[Any, dtype[np.float64]]) -> \
        List[List[ndarray]]:
    """
    This function converts the intensity values to reflectance values.
    For each photo we convert each band separately
    by collecting all the intensities of the panels for the given photo and band.
    The intensities are then combined with the known reflectance values of the panels
    at the given band to fit a linear function (Empirical Line Method).
    Read more about ELM: https://www.asprs.org/wp-content/uploads/2015/05/3E%5B5%5D-paper.pdf
    :param paths: List of orthophoto paths
    :param intensities: intensity values matrix of shape (photo, panel, band)
    :return: List of reflectance photos, each photo is a list of bands, each band is a ndarray of shape (width, height)
    """
    unconverted_photos = []
    converted_photos = []
    for photo_index, orthophoto_path in enumerate(tqdm(paths)):
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
    if len(unconverted_photos) > 0:
        print("WARNING: Could not convert", len(unconverted_photos), "photos")
    return converted_photos


def build_batches_per_full_visibility(paths_with_visibility: Dict[Path, ndarray]) -> List[List[Path]]:
    batches = []
    current_batch = []
    for (path, panel_visibility), nextVisibility in tqdm(get_next(paths_with_visibility.items())):
        current_batch.append(path)

        all_panels_visible = panel_visibility.all()
        if all_panels_visible:
            batches.append(current_batch)
            if nextVisibility is not None and nextVisibility[1].all():
                # The next image has all panels visible, so no need to include this one twice
                current_batch = []
            else:
                # This results in photos at the end of a batch being also in the next batch.
                # Therefore, they are calculated twice, but otherwise the interpolation would lose their information.
                # The next batch could start with an images without panels in it,
                # which would need the image with panels before it to be interpolated correctly
                current_batch = [path]
    batches.append(current_batch)
    return batches


def orthophoto_main():
    panel_locations = load_panel_locations(Path(args.panel_locations_file))
    orthophoto_paths = get_orthophoto_paths(args.dataset)

    photo: DatasetReader
    with rasterio.open(orthophoto_paths[0]) as photo:
        number_of_bands = len(photo.read())

    # --- Input validation
    if len(panel_properties) != len(panel_locations):
        raise Exception("Number of panel specifications does not match number of panel locations")
    if number_of_bands != len(panel_properties[0]['bands']):
        raise Exception("Number of bands in the images does not match number of bands in the panel specification")

    paths_with_visibility = {}
    for path in tqdm(orthophoto_paths):
        panels_visible = np.array(
            [is_panel_in_orthophoto(path, p) for p in panel_locations]
        )
        paths_with_visibility[path] = panels_visible

    batches = build_batches_per_full_visibility(paths_with_visibility)

    output_folder = None
    if args.debug:
        output_folder = args.dataset + "/debug/intensity/"
        for p in Path(output_folder).glob("*.csv"):
            p.unlink()

    for batch in batches:
        # --- Run pipeline
        i = extract_intensities_from_orthophotos(batch, paths_with_visibility, panel_locations, number_of_bands)
        if args.debug:
            debug_save_intensities(i, number_of_bands, output_folder)
        i = interpolate_intensities(i, number_of_bands, panel_properties)
        if args.debug:
            if args.debug:
                debug_save_intensities(i, number_of_bands, output_folder, "_interpolated")
        c = convert_orthophotos_to_reflectance(batch, i)
        del i
        save_orthophotos(batch, c)
        del c
    if args.debug:
        debug_combine_and_plot_intensities(len(orthophoto_paths), number_of_bands, output_folder, panel_properties)
        debug_combine_and_plot_intensities(len(orthophoto_paths), number_of_bands, output_folder, panel_properties,
                                           "_interpolated")


if __name__ == '__main__':
    # --- Get input arguments from user
    logging.basicConfig(level=logging.INFO)


    class GeolocationArgumentParser(Tap):
        dataset: str  # Path to the dataset folder
        panel_locations_file: str | None = None  # Path to file instead "geolocations.gpk" in the dataset folder
        panel_properties_file: str | None = None  # Path to file instead "panel_properties.json" in the dataset folder
        debug: bool = False  # Prints logs and adds debug images into a /debug/ directory in the dataset folder

        def configure(self):
            self.add_argument('dataset')
            self.add_argument('-d', '--debug')


    args = GeolocationArgumentParser(
        description='Automatically detect reflection calibration panels in images and transform the given images to '
                    'reflectance',
        epilog='If you have any questions, please contact').parse_args()

    # Input Validation
    if args.panel_locations_file is not None and not os.path.exists(args.panel_locations_file):
        raise Exception("Could not find specified panel location file: {}".format(args.panel_locations_file))

    if args.panel_properties_file is not None and not os.path.exists(args.panel_properties_file):
        raise Exception("Could not find specified panel properties file: {}".format(args.panel_properties_file))

    panel_properties = load_panel_properties(args.dataset, args.panel_properties_file)

    orthophoto_main()
