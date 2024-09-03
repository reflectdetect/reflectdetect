import os
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

from reflectdetect.PanelProperties import ApriltagPanelPropertiesFile, ValidatedApriltagPanelProperties, \
    validate_apriltag_panel_properties
from reflectdetect.constants import IMAGES_FOLDER, PANEL_PROPERTIES_FILENAME
from reflectdetect.pipeline import interpolate, convert, fit
from reflectdetect.utils.apriltags import detect_tags, get_altitude_from_panels, get_panel, get_detector_config, \
    save_images, build_batches_per_band
from reflectdetect.utils.debug import debug_combine_and_plot_intensities, debug_show_panel, \
    debug_save_intensities_single_band, ProgressBar
from reflectdetect.utils.exif import get_camera_properties
from reflectdetect.utils.panel import calculate_panel_size_in_pixels, get_band_reflectance
from reflectdetect.utils.paths import get_output_path
from reflectdetect.utils.polygons import shrink_or_swell_shapely_polygon
from reflectdetect.utils.thread import run_in_thread


class ApriltagArgumentParser(Tap):
    dataset: Path  # Path to the dataset folder
    panel_properties_file: str | None = None  # Path to file instead "panel_properties.json" in the dataset folder
    images_folder: str | None = None  # Path to images folder instead "/images" in the dataset folder
    default_panel_width: float | None = None  # Width of the calibration panel in meters
    default_panel_height: float | None = None  # Height of the calibration panel in meters
    tag_size: float | None = None  # Size of the apriltags in meters (Only measure the primary detection area, see apriltag_area_measurement.ipynb)
    default_tag_direction: str = "up"  # (up, down, left, right) Direction of the panel with respect to the tag. Down direction is where the text is printed on the tag
    default_tag_family: str = "tag25h9"  # Name of the apriltag family used
    default_shrink_factor: float = 0.8  # This factor gets multiplied to the detected panel area, to avoid artifacts like bleed
    default_panel_smudge_factor: float = 0.8  # This factor gets multiplied to the panel width and height to account for inaccuracy in lens exif information given by the manufacturer
    default_tag_smudge_factor: float = 1.0  # This factor gets multiplied to distance between tag and panel, useful if the tag was placed to far from the panel
    debug: bool = False  # Prints logs and adds debug images into a /debug/ directory in the dataset folder

    def configure(self) -> None:
        self.add_argument('dataset', nargs="?", default=".", type=Path)
        self.add_argument('-d', '--debug')


class AprilTagEngine:
    def __init__(self, args: ApriltagArgumentParser):
        self.progress = Progress(SpinnerColumn(),
                                 TextColumn("[progress.description]{task.description}", table_column=Column(width=40)),
                                 TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), BarColumn(),
                                 MofNCompleteColumn(), TextColumn("•"), TimeElapsedColumn(), )
        self.progress.start()
        self.dataset = Path(args.dataset) if args.dataset is not None else None
        self.debug = args.debug

        # Input Validation

        # Panel_properties file
        panel_properties_filepath = Path(args.panel_properties_file) if args.panel_properties_file is not None else None
        if panel_properties_filepath is not None:
            if not panel_properties_filepath.exists():
                raise Exception(f"Could not find specified panel properties file: {args.panel_properties_file}")
        else:
            if not (self.dataset / PANEL_PROPERTIES_FILENAME).exists():
                raise Exception(f"No panel properties file found at {self.dataset / PANEL_PROPERTIES_FILENAME}.")
        panel_properties_file = self.load_panel_properties(panel_properties_filepath)

        def default(value, default):
            return value if value is not None else default

        default_properties_names = ["default_panel_width", "default_panel_height"
                                                           "default_tag_family", "default_tag_direction",
                                    "default_panel_smudge_factor",
                                    "default_tag_smudge_factor", "default_shrink_factor"]
        default_properties = {}
        for prop_name in default_properties_names:
            # Get the default values for the panel properties first from the CLI args
            # and then from the panel_properties file.
            # If one of the defaults is still None
            # and one of the panels does not specify the value the default would be used for an Exception will be thrown
            # in the validate function below
            default_properties[prop_name] = default(getattr(args, prop_name), getattr(panel_properties_file, prop_name))

        self.panel_properties: list[ValidatedApriltagPanelProperties] = validate_apriltag_panel_properties(
            panel_properties_file.panel_properties, default_properties)

        self.tag_size = args.tag_size if args.tag_size is not None else panel_properties_file.tag_size
        if self.tag_size is None:
            raise Exception(f"Tag size not set via panel_properties file or CLI argument")

        self.number_of_panels = len(self.panel_properties)
        self.progress.console.print("Collected information of", self.number_of_panels, "panels") if self.debug else None

        images_folder = Path(args.images_folder) if args.images_folder is not None else None
        if images_folder is not None:
            if not images_folder.exists():
                raise Exception(f"Could not find specified images folder: {args.images_folder}")
        else:
            if not (self.dataset / IMAGES_FOLDER).exists():
                raise Exception(f"No images folder found at: {self.dataset / IMAGES_FOLDER}")

        self.images_paths = self.get_apriltag_paths(images_folder)
        self.progress.console.print("Found", len(self.images_paths), "images") if self.debug else None

        self.all_ids = list(set([p.tag_id for p in self.panel_properties]))

        self.progress.console.print("Loading detector...") if self.debug else None
        self.detector = AprilTagDetector()
        for family in [p.tag_family for p in self.panel_properties]:
            if not self.detector.addFamily(family):
                raise Exception("Apriltag Family not recognized:", family)
        self.detector.setConfig(get_detector_config())

    def load_panel_properties(self, panel_properties_file: Path | None) -> ApriltagPanelPropertiesFile:
        if panel_properties_file is None:
            path = self.dataset / PANEL_PROPERTIES_FILENAME
        else:
            path = panel_properties_file

        return ApriltagPanelPropertiesFile.parse_file(path)

    def convert_images_to_reflectance(self, paths: list[Path], intensities: NDArray[np.float64], band_index: int) -> \
            list[NDArray[np.float64] | None]:
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

        with ProgressBar(self.progress, description="Converting images", total=len(paths)) as pb:
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
                coefficients = fit(intensities_of_panels, get_band_reflectance(self.panel_properties, band_index))
                band = cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE)
                converted_photos.append(convert(band, coefficients))
                pb.update()
            if len(unconverted_photos) > 0:
                self.progress.console.print("[red] Could not convert", len(unconverted_photos), "photos")
            return converted_photos

    def extract_using_apriltags(self, path: Path) -> list[None | float]:
        img = cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE)

        all_tags = detect_tags(img, self.detector, self.all_ids)
        altitude = get_altitude_from_panels(all_tags, path, (len(img[0]), len(img)), self.tag_size)

        resolution = (len(img[0]), len(img))
        focal_length_mm, focal_plane_x_res, focal_plane_y_res, focal_plane_resolution_unit = get_camera_properties(path)
        panel_intensities: list[float | None] = [None] * self.number_of_panels
        debug_corners = []
        debug_tags = []
        for tag in all_tags:
            # TODO remove family from filter
            panels = list(
                filter(lambda p: p.tag_id == tag.getId(), self.panel_properties))
            if not len(panels) == 1:
                raise Exception("Could not associate panel with found tag")
            panel_index = self.panel_properties.index(panels[0])
            panel: ValidatedApriltagPanelProperties = panels[0]
            panel_size_pixel = calculate_panel_size_in_pixels(altitude, resolution,
                                                              (panel.panel_width, panel.panel_height),
                                                              focal_length_mm,
                                                              focal_plane_x_res, focal_plane_y_res,
                                                              focal_plane_resolution_unit, panel.panel_smudge_factor)
            corners = get_panel(tag, panel_size_pixel, resolution, panel.tag_smudge_factor, panel.tag_direction)

            if corners is None:
                continue
            else:
                if self.debug:
                    debug_corners.append(corners)
                    debug_tags.append(tag)
                polygon = shapely.Polygon(corners)
                polygon = shrink_or_swell_shapely_polygon(polygon, 1.0-panel.shrink_factor)
                panel_mask = rasterize([polygon], out_shape=img.shape)
                mean: float = float(np.ma.MaskedArray(img, mask=~(panel_mask.astype(np.bool_))).mean())  # type: ignore
                panel_intensities[panel_index] = mean
        if self.debug:
            output_path = get_output_path(self.dataset, path, "panels.tif", "debug/panels")
            run_in_thread(debug_show_panel, True, img, debug_tags, debug_corners, output_path)
        return panel_intensities

    def extract_intensities_from_apriltags(self, batch: list[Path]) -> NDArray[np.float64]:
        with ProgressBar(self.progress, "Extracting intensities", total=len(batch)) as pb:
            intensities = np.zeros((len(batch), self.number_of_panels))
            for img_index, path in enumerate(batch):
                panel_intensities = self.extract_using_apriltags(path)
                intensities[img_index] = panel_intensities
                pb.update()
            return intensities

    def get_apriltag_paths(self, images_folder: Path | None) -> list[Path]:
        if images_folder is None:
            path = self.dataset / IMAGES_FOLDER
            if not path.exists():
                raise ValueError(f"No images folder found at {path}.")
        else:
            path = images_folder

        return list(sorted([filepath for filepath in path.glob("*.tif")]))

    def start(self) -> None:
        all_images_task = self.progress.add_task("Total Progress", total=len(self.images_paths))
        batches = build_batches_per_band(self.images_paths)
        number_of_bands = len(batches)
        self.progress.console.print("Processing", number_of_bands, "bands") if self.debug else None

        debug_output_folder = self.dataset / "debug"
        if self.debug:
            for p in (debug_output_folder / "intensity").glob("*.csv"):
                p.unlink()
            for p in (debug_output_folder / "panels").glob("*.tif"):
                p.unlink()

        for band_index, batch in enumerate(batches):
            i = self.extract_intensities_from_apriltags(batch)
            os.system("cls||clear")

            debug_save_intensities_single_band(i, band_index, debug_output_folder / "intensity") if self.debug else None
            with ProgressBar(self.progress, "Interpolating intensities", self.number_of_panels) as pb:
                for panel_index, _ in enumerate(self.panel_properties):
                    i[:, panel_index] = interpolate(i[:, panel_index])
                    pb.update()
            debug_save_intensities_single_band(i, band_index, debug_output_folder / "intensity",
                                               "_interpolated") if self.debug else None
            c = self.convert_images_to_reflectance(batch, i, band_index)
            del i
            save_images(self.dataset, batch, c, self.progress)
            del c
            self.progress.update(all_images_task, advance=len(batch))
        if self.debug:
            number_of_image_per_band = int(len(self.images_paths) / number_of_bands)
            debug_combine_and_plot_intensities(number_of_image_per_band, number_of_bands, self.number_of_panels,
                                               debug_output_folder / "intensity", )
            debug_combine_and_plot_intensities(number_of_image_per_band, number_of_bands, self.number_of_panels,
                                               debug_output_folder / "intensity", "_interpolated")


def main() -> None:
    # --- Get input arguments from user
    args = ApriltagArgumentParser(
        description='Automatically detect reflection calibration panels in images and transform the given images to '
                    'reflectance', epilog='If you have any questions, please contact').parse_args()

    AprilTagEngine(args).start()


if __name__ == '__main__':
    main()
