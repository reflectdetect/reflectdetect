import logging
import os
import re
from pathlib import Path

import cv2
import numpy as np
import shapely
from numpy._typing import NDArray
from pydantic import BaseModel
from pydantic.v1 import parse_raw_as
from rasterio.features import rasterize
from robotpy_apriltag import AprilTagDetector
from tap import Tap
from tqdm import tqdm

from pipeline import interpolate, convert, fit
from reflectdetect.utils.exif import get_camera_properties
from utils.apriltags import detect_tags, get_altitude_from_panels, get_panel, get_detector_config, save_images
from utils.debug import debug_combine_and_plot_intensities, debug_show_panel, debug_save_intensities_single_band
from utils.panel import calculate_panel_size_in_pixels, get_band_reflectance
from utils.paths import get_output_path
from utils.polygons import shrink_or_swell_shapely_polygon

logger = logging.getLogger(__name__)


class ApriltagPanelProperties(BaseModel):
    bands: list[float]
    tag_id: int
    family: str  # TODO: remove


def load_panel_properties(dataset: Path, panel_properties_file: Path | None) -> list[ApriltagPanelProperties]:
    if panel_properties_file is None:
        canonical_filename = "panels_properties.json"
        path = dataset / canonical_filename
        if not os.path.exists(path):
            raise ValueError("No panel properties file found at {}.".format(path))
    else:
        path = panel_properties_file

    with open(path) as f:
        json_string: str = f.read()
        panels = parse_raw_as(list[ApriltagPanelProperties], json_string)
    return panels


def convert_images_to_reflectance(paths: list[Path], intensities: NDArray[np.float64], band_index: int) -> list[
    NDArray[np.float64] | None]:
    """
    This function converts the intensity values to reflectance values.
    For each photo we convert each band separately
    by collecting all the intensities of the panels for the given photo and band.
    The intensities are then combined with the known reflectance values of the panels
    at the given band to fit a linear function (Empirical Line Method).
    Read more about ELM: https://www.asprs.org/wp-content/uploads/2015/05/3E%5B5%5D-paper.pdf
    :param band_index: index of the band of the image
    :param paths: list of image paths
    :param intensities: intensity values matrix of shape (photo, panel, band)
    :return: list of reflectance photos, each photo is a list of bands, each band is a ndarray of shape (width, height)
    """
    unconverted_photos = []
    converted_photos: list[NDArray[np.float64] | None] = []
    for image_index, path in enumerate(tqdm(paths)):
        intensities_of_panels = intensities[image_index, :]
        if np.isnan(intensities_of_panels).any():
            # If for some reason not all intensities are present, we save the indices for debugging purposes
            # A None value is appended to the converted photos to not lose
            # the connection between orthophotos and converted photos based on the index
            unconverted_photos.append(image_index)
            converted_photos.append(None)
            break
        coefficients = fit(intensities_of_panels, get_band_reflectance(panel_properties, band_index))
        band = cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE)
        converted_photos.append(convert(band, coefficients))

    if len(unconverted_photos) > 0:
        print("WARNING: Could not convert", len(unconverted_photos), "photos")
    return converted_photos


def extract_using_apriltags(path: Path, detector: AprilTagDetector, all_ids: list[int],
                            panel_size_m: tuple[float, float], tag_size_m: float) -> list[None | float]:
    img = cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE)

    all_tags = detect_tags(img, detector, all_ids)
    if len(all_tags) != len(panel_properties):
        return [None] * len(panel_properties)

    altitude = get_altitude_from_panels(all_tags, path, (len(img[0]), len(img)), tag_size_m)

    resolution = (len(img[0]), len(img))
    properties = get_camera_properties(path)
    panel_size_pixel = calculate_panel_size_in_pixels(altitude, resolution, panel_size_m, *properties)
    panel_intensities: list[float | None] = [None] * len(panel_properties)
    for tag in all_tags:
        panels = list(filter(lambda p: p.family == tag.getFamily() and p.tag_id == tag.getId(), panel_properties))
        if not len(panels) == 1:
            raise Exception("Could not associate panel with found tag")
        panel_index = panel_properties.index(panels[0])
        corners = get_panel(tag, panel_size_pixel[0], resolution)

        if corners is None:
            continue
        else:
            if args.debug:
                output_path = get_output_path(path, "panel_" + str(tag.getId()) + "_" + tag.getFamily(), "debug/panels")
                debug_show_panel(img, [tag], corners, output_path)
            polygon = shapely.Polygon(corners)
            polygon = shrink_or_swell_shapely_polygon(polygon, 0.2)
            panel_mask = rasterize([polygon], out_shape=img.shape)
            mean: float = float(np.ma.MaskedArray(img, mask=~(panel_mask.astype(np.bool_))).mean())  # type: ignore
            panel_intensities[panel_index] = mean
    return panel_intensities


def extract_intensities_from_apriltags(batch: list[Path], detector: AprilTagDetector, all_ids: list[int],
                                       panel_size_m: tuple[float, float], tag_size_m: float) -> NDArray[np.float64]:
    intensities = np.zeros((len(batch), len(panel_properties)))
    for img_index, path in enumerate(tqdm(batch)):
        panel_intensities = extract_using_apriltags(path, detector, all_ids, panel_size_m, tag_size_m)
        intensities[img_index] = panel_intensities
    return intensities


def get_apriltag_paths(dataset: str) -> list[Path]:
    folder = "images"
    template_path = Path(dataset + "/" + folder).resolve()
    return list(sorted([filepath for filepath in template_path.glob("*.tif")]))


def build_batches_per_band(paths: list[Path]) -> list[list[Path]]:
    # TODO also add visibility batching
    batches: list[list[Path]] = []
    # Regular expression to match file names and capture the base and suffix
    pattern = re.compile(r".*_(\d+)\.tif$")

    for image_path in paths:
        match = pattern.match(image_path.name)
        if match:
            band_index = int(match.group(1)) - 1
            if band_index == len(batches):
                batches.append([])
            elif band_index > len(batches) or band_index < 0:
                raise ValueError(
                    "Problem with the sorting of the pats or regex")  # TODO better path parsing generalization
            batches[band_index].append(image_path)

    return batches


def apriltag_main() -> None:
    img_paths = get_apriltag_paths(args.dataset)
    batches = build_batches_per_band(img_paths)
    number_of_bands = len(batches)

    # TODO: add to args
    tag_size_m = 0.3
    panel_size_m = (0.8, 0.8)

    d = AprilTagDetector()
    d.addFamily(args.family)
    detector_config = get_detector_config()
    d.setConfig(detector_config)

    all_ids = list(set([p.tag_id for p in panel_properties]))

    # Hack TODO: remove, as panels should not specify family
    d = AprilTagDetector()
    for family in [p.family for p in panel_properties]:
        d.addFamily(family)
    d.setConfig(detector_config)

    output_folder = Path(args.dataset) / "debug"
    if args.debug:
        for p in (output_folder / "intensity").glob("*.csv"):
            p.unlink()
        for p in (output_folder / "panels").glob("*.tif"):
            p.unlink()

    for (band_index, batch) in enumerate(batches):
        logger.info("Processing batch for band", band_index, "with length", len(batch))
        # --- Run pipeline
        i = extract_intensities_from_apriltags(batch, d, all_ids, panel_size_m, tag_size_m)
        if args.debug:
            debug_save_intensities_single_band(i, band_index, output_folder / "intensity")
        for panel_index, _ in enumerate(panel_properties):
            i[:, panel_index] = interpolate(i[:, panel_index])
        if args.debug:
            debug_save_intensities_single_band(i, band_index, output_folder / "intensity", "_interpolated")
        c = convert_images_to_reflectance(batch, i, band_index)
        del i
        save_images(batch, c)
        del c
    if args.debug:
        number_of_image_per_band = int(len(img_paths) / number_of_bands)
        debug_combine_and_plot_intensities(number_of_image_per_band, number_of_bands, len(panel_properties),
                                           output_folder / "intensity", )
        debug_combine_and_plot_intensities(number_of_image_per_band, number_of_bands, len(panel_properties),
                                           output_folder / "intensity", "_interpolated")


if __name__ == '__main__':
    # --- Get input arguments from user
    logging.basicConfig(level=logging.INFO)


    class ApriltagArgumentParser(Tap):
        dataset: str  # Path to the dataset folder
        family: str  # Name of the apriltag family used
        panel_properties_file: str | None = None  # Path to file instead "panel_properties.json" in the dataset folder
        debug: bool = False  # Prints logs and adds debug images into a /debug/ directory in the dataset folder

        def configure(self) -> None:
            self.add_argument('dataset')
            self.add_argument('-d', '--debug')


    args = ApriltagArgumentParser(
        description='Automatically detect reflection calibration panels in images and transform the given images to '
                    'reflectance', epilog='If you have any questions, please contact').parse_args()

    if args.panel_properties_file is not None and not os.path.exists(args.panel_properties_file):
        raise Exception("Could not find specified panel properties file: {}".format(args.panel_properties_file))

    panel_properties_file = Path(args.panel_properties_file) if args.panel_properties_file is not None else None
    dataset = Path(args.dataset) if args.dataset is not None else None
    panel_properties = load_panel_properties(dataset, panel_properties_file)

    # Input Validation##TODO

    test_detector = AprilTagDetector()
    if not test_detector.addFamily(args.family):
        raise Exception("Family not recognized")

    apriltag_main()