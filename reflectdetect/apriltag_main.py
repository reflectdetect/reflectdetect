import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import shapely
from numpy.typing import NDArray
from rasterio.features import rasterize
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, SpinnerColumn
from rich.table import Column
from robotpy_apriltag import AprilTagDetector
from tap import Tap

from reflectdetect.PanelProperties import ApriltagPanelProperties
from reflectdetect.constants import IMAGES_FOLDER, PANEL_PROPERTIES_FILENAME
from reflectdetect.pipeline import interpolate, convert, fit
from reflectdetect.utils.apriltags import detect_tags, get_altitude_from_panels, get_panel, get_detector_config, \
    save_images
from reflectdetect.utils.debug import debug_combine_and_plot_intensities, debug_show_panel, \
    debug_save_intensities_single_band, ProgressBar
from reflectdetect.utils.exif import get_camera_properties
from reflectdetect.utils.panel import calculate_panel_size_in_pixels, get_band_reflectance
from reflectdetect.utils.paths import get_output_path
from reflectdetect.utils.polygons import shrink_or_swell_shapely_polygon
from reflectdetect.utils.thread import run_in_thread


def load_panel_properties(dataset: Path, panel_properties_file: Path | None) -> list[ApriltagPanelProperties]:
    if panel_properties_file is None:
        path = dataset / PANEL_PROPERTIES_FILENAME
    else:
        path = panel_properties_file

    with open(path) as f:
        panels = [ApriltagPanelProperties.parse_obj(item) for item in json.load(f)]
    return panels


def convert_images_to_reflectance(paths: list[Path], intensities: NDArray[np.float64], band_index: int,
                                  progress: Progress | None = None) -> list[NDArray[np.float64] | None]:
    """
    This function converts the intensity values to reflectance values.
    For each photo we convert each band separately
    by collecting all the intensities of the panels for the given photo and band.
    The intensities are then combined with the known reflectance values of the panels
    at the given band to fit a linear function (Empirical Line Method).
    Read more about ELM: https://www.asprs.org/wp-content/uploads/2015/05/3E%5B5%5D-paper.pdf
    :param progress:
    :param band_index: index of the band of the image
    :param paths: list of image paths
    :param intensities: intensity values matrix of shape (photo, panel, band)
    :return: list of reflectance photos, each photo is a list of bands, each band is a ndarray of shape (width, height)
    """

    with ProgressBar(progress, description="Converting images", total=len(paths)) as pb:
        unconverted_photos = []
        converted_photos: list[NDArray[np.float64] | None] = []
        for image_index, path in enumerate(paths):
            intensities_of_panels = intensities[image_index, :]
            if np.isnan(intensities_of_panels).any():
                # If for some reason not all intensities are present, we save the indices for debugging purposes
                # A None value is appended to the converted photos to not lose
                # the connection between orthophotos and converted photos based on the index
                unconverted_photos.append(image_index)
                converted_photos.append(None)
                pb.update()
                continue
            coefficients = fit(intensities_of_panels, get_band_reflectance(panel_properties, band_index))
            band = cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE)
            converted_photos.append(convert(band, coefficients))
            pb.update()
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
    focal_length_mm, focal_plane_x_res, focal_plane_y_res, focal_plane_resolution_unit = get_camera_properties(path)
    panel_size_pixel = calculate_panel_size_in_pixels(altitude, resolution, panel_size_m, focal_length_mm,
                                                      focal_plane_x_res, focal_plane_y_res, focal_plane_resolution_unit,
                                                      args.panel_smudge_factor)
    panel_intensities: list[float | None] = [None] * len(panel_properties)
    for tag in all_tags:
        # TODO remove family from filter
        panels = list(filter(lambda p: p.family == tag.getFamily() and p.tag_id == tag.getId(), panel_properties))
        if not len(panels) == 1:
            raise Exception("Could not associate panel with found tag")
        panel_index = panel_properties.index(panels[0])
        corners = get_panel(tag, panel_size_pixel, resolution, args.tag_smudge_factor)

        if corners is None:
            continue
        else:
            if args.debug:
                output_path = get_output_path(path, "panel_" + str(tag.getId()) + "_" + tag.getFamily(), "debug/panels")
                run_in_thread(debug_show_panel, img, tag, corners, args.shrink_factor, output_path)
            polygon = shapely.Polygon(corners)
            polygon = shrink_or_swell_shapely_polygon(polygon, args.shrink_factor)
            panel_mask = rasterize([polygon], out_shape=img.shape)
            mean: float = float(np.ma.MaskedArray(img, mask=~(panel_mask.astype(np.bool_))).mean())  # type: ignore
            panel_intensities[panel_index] = mean
    return panel_intensities


def extract_intensities_from_apriltags(batch: list[Path], detector: AprilTagDetector, all_ids: list[int],
                                       panel_size_m: tuple[float, float], tag_size_m: float,
                                       progress: Progress | None = None) -> NDArray[np.float64]:
    with ProgressBar(progress, "Extracting intensities", total=len(batch)) as pb:
        intensities = np.zeros((len(batch), len(panel_properties)))
        for img_index, path in enumerate(batch):
            panel_intensities = extract_using_apriltags(path, detector, all_ids, panel_size_m, tag_size_m)
            intensities[img_index] = panel_intensities
            pb.update()
        return intensities


def get_apriltag_paths(dataset: Path, images_folder: Path | None) -> list[Path]:
    if images_folder is None:
        path = dataset / IMAGES_FOLDER
        if not path.exists():
            raise ValueError(f"No images folder found at {path}.")
    else:
        path = images_folder

    return list(sorted([filepath for filepath in path.glob("*.tif")]))


def build_batches_per_band(paths: list[Path]) -> list[list[Path]]:
    # TODO also add visibility batching
    batches: list[list[Path]] = []
    # Regular expression to match file names and capture the base and suffix
    pattern = re.compile(r".*_(\d+)\.tif$")  # TODO better path parsing generalization

    for image_path in paths:
        match = pattern.match(image_path.name)
        if not match:
            raise Exception(f"Could not extract band index from filename")
        band_index = int(match.group(1)) - 1
        if band_index > len(batches) or band_index < 0:
            raise Exception("Problem with the sorting of the pats or regex")
        if band_index == len(batches):
            batches.append([])
        batches[band_index].append(image_path)
    return batches


def apriltag_main(dataset: Path, images_folder: Path | None, debug: bool = False) -> None:
    with Progress(SpinnerColumn(),
                  TextColumn("[progress.description]{task.description}", table_column=Column(width=40)),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), BarColumn(), MofNCompleteColumn(),
                  TextColumn("•"), TimeElapsedColumn(),
                  ) as progress:
        img_paths = get_apriltag_paths(dataset, images_folder)
        os.system("cls||clear")
        all_images_task = progress.add_task("Total Progress", total=len(img_paths))

        progress.console.print("Batching...") if debug else None
        batches = build_batches_per_band(img_paths)
        number_of_bands = len(batches)

        progress.console.print("Loading detector...") if debug else None
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

        output_folder = dataset / "debug"
        if debug:
            for p in (output_folder / "intensity").glob("*.csv"):
                p.unlink()
            for p in (output_folder / "panels").glob("*.tif"):
                p.unlink()

        for band_index, batch in enumerate(batches):
            i = extract_intensities_from_apriltags(batch, d, all_ids, panel_size_m, args.tag_size, progress)
            os.system("cls||clear")
            if debug:
                debug_save_intensities_single_band(i, band_index, output_folder / "intensity")
            interpolate_task = progress.add_task(description="Interpolating intensities", total=len(panel_properties))
            for panel_index, _ in enumerate(panel_properties):
                i[:, panel_index] = interpolate(i[:, panel_index])
                progress.update(interpolate_task, advance=1)
            progress.remove_task(interpolate_task)
            if debug:
                debug_save_intensities_single_band(i, band_index, output_folder / "intensity", "_interpolated")
            c = convert_images_to_reflectance(batch, i, band_index, progress)
            del i
            save_images(batch, c, progress)
            del c
            progress.update(all_images_task, advance=len(batch))
        if debug:
            number_of_image_per_band = int(len(img_paths) / number_of_bands)
            debug_combine_and_plot_intensities(number_of_image_per_band, number_of_bands, len(panel_properties),
                                               output_folder / "intensity", )
            debug_combine_and_plot_intensities(number_of_image_per_band, number_of_bands, len(panel_properties),
                                               output_folder / "intensity", "_interpolated")


if __name__ == '__main__':
    # --- Get input arguments from user

    class ApriltagArgumentParser(Tap):
        dataset: str  # Path to the dataset folder
        family: str  # Name of the apriltag family used
        panel_properties_file: str | None = None  # Path to file instead "panel_properties.json" in the dataset folder
        debug: bool = False  # Prints logs and adds debug images into a /debug/ directory in the dataset folder
        images_folder: str | None = None  # Path to images folder instead "/images" in the dataset folder
        shrink_factor: float = 0.2  # How many percent to shrink the detected panel area, to avoid artifacts like bleed
        panel_smudge_factor: float = 1.0  # This factor gets multiplied to the panel width and height to account for inaccuracy in lens exif information given by the manufacturer
        tag_smudge_factor: float = 1.0  # This factor gets multiplied to the panel width and height to account for inaccuracy in lens exif information given by the manufacturer
        tag_size: float  # Size of the apriltags in meters (Only measure the primary detection area, see apriltag_primary.ipynb)
        panel_width: float
        panel_height: float

        def configure(self) -> None:
            self.add_argument('dataset')
            self.add_argument('-d', '--debug')


    args = ApriltagArgumentParser(
        description='Automatically detect reflection calibration panels in images and transform the given images to '
                    'reflectance', epilog='If you have any questions, please contact').parse_args()

    panel_size_m = (args.panel_width, args.panel_height)
    panel_properties_file = Path(args.panel_properties_file) if args.panel_properties_file is not None else None
    images_folder = Path(args.images_folder) if args.images_folder is not None else None
    dataset = Path(args.dataset) if args.dataset is not None else None
    panel_properties = load_panel_properties(dataset, panel_properties_file)

    # Input Validation##TODO
    if panel_properties_file is not None:
        if not panel_properties_file.exists():
            raise Exception(f"Could not find specified panel properties file: {args.panel_properties_file}")
    else:
        if not (dataset / PANEL_PROPERTIES_FILENAME).exists():
            raise Exception(f"No panel properties file found at {dataset / PANEL_PROPERTIES_FILENAME}.")

    if images_folder is not None:
        if not images_folder.exists():
            raise Exception(f"Could not find specified images folder: {args.images_folder}")
    else:
        if not (dataset / IMAGES_FOLDER).exists():
            raise Exception(f"Could not find specified images folder: {args.images_folder}")

    test_detector = AprilTagDetector()
    if not test_detector.addFamily(args.family):
        raise Exception("Apriltag Family not recognized")

    apriltag_main(dataset, images_folder, args.debug)
