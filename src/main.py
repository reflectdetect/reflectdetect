import argparse
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit
from rasterio import DatasetReader
from robotpy_apriltag import AprilTagDetector
from tqdm import tqdm

from orthophoto_utils import load_panel_properties, load_panel_locations, get_orthophoto_paths, extract, interpolate, \
    fit, get_band_reflectance, convert, save, is_panel_in_orthophoto
from utils.iterators import get_next

logger = logging.getLogger(__name__)


def extract_intensities(batch_of_orthophotos: list[tuple[Path, list[bool]]]) -> ndarray[
    Any, dtype[floating[_64Bit] | float_]]:
    """
    This function extracts intensities from the orthophotos.
    It does so by looking at each photo and determining which panels are visible in the photo.
    If a panel is not visible in the image it is not visible in any of the bands
    as the photos are assumed to rectified and aligned.
    Therefore, extraction is skipped for that panel with `np.Nan` values saved in the output.
    Otherwise, the function uses the recorded gps location of the panel given by `panel_locations`
    to extract the mean intensity of the panel in that band for that photo
    :return: The extracted intensities for all orthophotos, with np.Nan for values that could not be found.
    """
    intensities = np.zeros((len(batch_of_orthophotos), len(panel_locations), number_of_bands))
    for photo_index, (orthophoto_path, panel_occurrence) in enumerate(tqdm(batch_of_orthophotos)):
        for panel_index, panel_location in enumerate(panel_locations):
            if not panel_occurrence[panel_index]:
                intensities[photo_index][panel_index] = np.full(number_of_bands, np.NaN)
                continue
            # extract the mean intensity for each band at that panel location
            orthophoto: DatasetReader
            with rasterio.open(orthophoto_path) as orthophoto:
                panel_intensities_per_band = extract(orthophoto, panel_location)
            intensities[photo_index][panel_index] = panel_intensities_per_band
    return intensities


def interpolate_intensities(intensities: ndarray[Any, dtype[floating[_64Bit] | float_]]) -> ndarray[
    Any, dtype[floating[_64Bit] | float_]]:
    """
    This function is used to piecewise linearly interpolate the intensity values to fill the `np.Nan` gaps in the data.
    To interpolate we select all the values captured in all the images for a given panel and band.
    Only for photos where the panel was visible we have a value for the given band.
    8Bit Data might look like this: [np.NaN, np.NaN, 240.0, 241.0, 240.0, np.NaN, 242.0, np.NaN, np.NaN]
    After interpolation: [240.0, 240.0, 240.0, 241.0, 240.0, 241.0, 240.0, 240.0, 240.0]
    :rtype: ndarray[Any, dtype[floating[_64Bit] | float_]]
    :param intensities: intensity values matrix of shape (photo, panel, band) with some values being np.NaN.
    :return: The interpolated intensity values
    """
    for panel_index, panel in enumerate(panel_locations):
        for band_index in range(0, number_of_bands):
            intensities[:, panel_index, band_index] = interpolate(intensities[:, panel_index, band_index])
    return intensities


def convert_photos_to_reflectance(paths: list[Path], intensities: ndarray[Any, dtype[floating[_64Bit] | float_]]) -> \
        list[list[ndarray]]:
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
            photo: ndarray = orthophoto.read()
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


def save_photos(paths: list[Path], converted_photos: list[list[ndarray]]) -> None:
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
        save(output_path, photo, meta)


def build_batches(orthophoto_paths: list[Path]) -> list[(list[Path], dict[Path, list[bool]])]:
    batches = []
    current_batch = []
    for path, next_path in tqdm(get_next(orthophoto_paths)):
        panels_visible = np.array(
            [is_panel_in_orthophoto(path, panel) for panel in panel_locations]
        )
        current_batch.append((path, panels_visible))

        all_panels_visible = panels_visible.sum() == len(panel_locations)
        if all_panels_visible:
            batches.append(current_batch)

            if np.array([is_panel_in_orthophoto(next_path, panel) for panel in panel_locations]).sum() == len(
                    panel_locations):
                current_batch = []
            else:
                # This results in photos at the end of a batch being also in the next batch.
                # Therefore they are calculated twice, but otherwise the interpolation would lose their information.
                # The next batch could start with a images without panels in it,
                # which would need the image with panels before it to be interpolated correctly
                current_batch = [(path, panels_visible)]
    batches.append(current_batch)
    return batches


def orthophoto_main():
    # python src/main.py "data/20240529_uav_multispectral_orthos_20m/orthophotos" "reflectance_panel_example_data.json" "data/20240529_uav_multispectral_orthos_20m/20240529_tarps_locations.gpkg"

    panel_properties = load_panel_properties(args.panel_properties)
    panel_locations = load_panel_locations(Path(args.panel_locations))
    orthophoto_paths = list(sorted(get_orthophoto_paths(args.images)))

    ophoto: DatasetReader
    with rasterio.open(orthophoto_paths[0]) as ophoto:
        number_of_bands = len(ophoto.read())

    # --- Input validation
    assert len(panel_properties) == len(panel_locations)
    assert number_of_bands == len(panel_properties[0]['bands'])

    batches = build_batches(orthophoto_paths)

    for batch in batches:
        # --- Run pipeline
        i = extract_intensities(batch)
        i = interpolate_intensities(i)
        paths = [path for (path, _) in batch]
        c = convert_photos_to_reflectance(paths, i)
        del i
        save_photos(paths, c)
        del c
        del paths


def apriltag_main():
    d = AprilTagDetector()
    d.addFamily(args.family)
    config = AprilTagDetector.Config()
    config.quadDecimate = 1.0
    config.numThreads = 4
    config.refineEdges = 1.0
    d.setConfig(config)

    for path in args.images:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if args.detection_type == 'apriltag_edges':
        raise NotImplementedError
    if args.detection_type == 'apriltag_single':
        raise NotImplementedError


if __name__ == '__main__':
    # --- Get input arguments from user
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog='ReflectDetect',
        description='Automatically detect reflection calibration panels in images and transform the given images to '
                    'reflectance',
        epilog='If you have any questions, please contact')
    parser.add_argument("images", help="Path to the image files", type=str)
    parser.add_argument("detection_type", help="Type of detection ('apriltag_edges', 'apriltag_single', 'geolocation')",
                        type=str)
    parser.add_argument("panel_properties", help="Path to the property file of the panels", type=str)
    parser.add_argument("panel_locations", help="Path to the GeoPackage file", type=str)
    parser.add_argument("family", help="Name of the apriltag family ('tag25h9', 'tagStandard41h12', ...)", type=str)
    args = parser.parse_args()

    detection_type: str = args.detection_type

    if detection_type not in ['apriltag_edges', 'apriltag_single', 'geolocation']:
        raise ValueError('detection_type', detection_type, 'unknown')

    if detection_type.startswith("apriltag_"):
        apriltag_main()
    if detection_type == 'geolocation':
        orthophoto_main()
