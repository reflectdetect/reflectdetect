import logging
import warnings
from pathlib import Path

import numpy as np
import rasterio
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from rasterio import DatasetReader, logging as rasterio_logging
from rich.progress import (
    Progress,
    TextColumn,
    TimeElapsedColumn,
    BarColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)
from rich.table import Column
from rich.traceback import install
from rich_argparse import RichHelpFormatter
from tap import Tap

import reflectdetect
from reflectdetect.PanelProperties import (
    GeolocationPanelPropertiesFile,
    ValidatedGeolocationPanelProperties,
    validate_geolocation_panel_properties,
)
from reflectdetect.constants import (
    PANEL_PROPERTIES_FILENAME,
    ORTHOPHOTO_FOLDER,
    DEFAULT_SHRINK_FACTOR,
    DEFAULT_PANEL_SMUDGE_FACTOR,
)
from reflectdetect.pipeline import interpolate_intensities, fit, convert
from reflectdetect.utils.debug import (
    debug_combine_and_plot_intensities,
    debug_save_intensities,
    ProgressBar,
    debug_show_geolocation,
)
from reflectdetect.utils.orthophoto import (
    load_panel_locations,
    get_orthophoto_paths,
    is_panel_in_orthophoto,
    save_orthophotos,
    build_batches_per_full_visibility,
    extract_using_geolocation,
)
from reflectdetect.utils.panel import get_band_reflectance
from reflectdetect.utils.paths import get_output_path, default, is_tool_installed
from reflectdetect.utils.thread import run_in_thread

logger = logging.getLogger(__name__)

rasterio_logging.disable()


class GeolocationArgumentParser(Tap):
    dataset: Path  # Path to the dataset folder
    panel_locations_file: str | None = None  # Path to file instead "geolocations.gpk" in the dataset folder
    panel_properties_file: str | None = None  # Path to file instead "panel_properties.json" in the dataset folder
    orthophotos_folder: str | None = None  # Path to orthophotos folder instead "/orthophotos" in the dataset folder
    debug: bool = False  # Prints logs and adds debug images into a /debug/ directory in the dataset folder
    default_shrink_factor: float = DEFAULT_SHRINK_FACTOR  # This factor gets multiplied to the detected panel area, to avoid artifacts like bleed
    default_panel_smudge_factor: float = DEFAULT_PANEL_SMUDGE_FACTOR  # This factor gets multiplied to the panel width and height to account for inaccuracy in lens exif information given by the manufacturer
    default_panel_width: float | None = None  # Width of the calibration panel in meters
    default_panel_height: float | None = None  # Height of the calibration panel in meters
    no_data_value: int = 0  # 65535  # The value in the tiff image to interpret as no-data
    debug_dpi: int | None = None  # Overwrite the default dpi debug images are generated at

    def configure(self) -> None:
        self.add_argument("dataset", nargs="?", default=".", type=Path)
        self.add_argument("-d", "--debug")


class GeolocationEngine:
    def __init__(self, args: GeolocationArgumentParser) -> None:
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn(
                "[self.progress.description]{task.description}",
                table_column=Column(width=40),
            ),
            TextColumn("[self.progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        )
        self.progress.start()
        self.dataset = Path(args.dataset) if args.dataset is not None else None
        if not self.dataset.exists():
            raise Exception(f"Could not find specified dataset folder: {args.dataset}")
        self.debug = args.debug
        self.debug_dpi = args.debug_dpi
        self.no_data_value = args.no_data_value
        # Validate Panel_properties file
        self.panel_properties: list[ValidatedGeolocationPanelProperties] = (
            self.validate_panel_properties(args)
        )
        self.number_of_panels = len(self.panel_properties)
        self.progress.console.print(
            "Collected information of", self.number_of_panels, "panels"
        ) if self.debug else None

        # Validate dataset
        orthophoto_folder = (
            Path(args.orthophotos_folder)
            if args.orthophotos_folder is not None
            else None
        )
        if orthophoto_folder is not None:
            if not orthophoto_folder.exists():
                raise Exception(
                    f"Could not find specified orthophoto folder: {args.orthophotos_folder}"
                )
        else:
            if not (self.dataset / ORTHOPHOTO_FOLDER).exists():
                raise Exception(
                    f"No orthophotos folder found at: {self.dataset / ORTHOPHOTO_FOLDER}"
                )
        self.orthophoto_paths = get_orthophoto_paths(self.dataset, orthophoto_folder)
        photo: DatasetReader
        with rasterio.open(self.orthophoto_paths[0]) as photo:
            self.number_of_bands = len(photo.read())
        self.number_of_photos = len(self.orthophoto_paths)
        self.progress.console.print(
            "Found", self.number_of_photos, "photos"
        ) if self.debug else None

        for index, panel in enumerate(self.panel_properties):
            if len(panel.bands) != self.number_of_bands:
                raise Exception(
                    f"Panel {index}: Number of bands does not match number of bands in the panel specification"
                )

        # Validate locations file
        panel_locations_file = (
            Path(args.panel_locations_file)
            if args.panel_locations_file is not None
            else None
        )
        if panel_locations_file is not None:
            if not panel_locations_file.exists():
                raise Exception(
                    f"Could not find specified panel location file: {args.panel_locations_file}"
                )
        else:
            if not (self.dataset / PANEL_PROPERTIES_FILENAME).exists():
                raise Exception(
                    f"Could not find file: {(self.dataset / PANEL_PROPERTIES_FILENAME)}"
                )
        with warnings.catch_warnings():
            # Ignore BadApplicationID Warning (action: "once" does not seem to work)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            panel_locations = load_panel_locations(self.dataset, panel_locations_file)

        # Validate connection of locations and properties
        panel_properties_layer_names = [
            panel.layer_name for panel in self.panel_properties
        ]
        panel_location_layer_names = [location[0] for location in panel_locations]
        if set(panel_location_layer_names).intersection(set(panel_properties_layer_names)) != set(
                panel_properties_layer_names):
            raise Exception(
                f"Panel properties layer names do not match panel locations layer names: location_file: {panel_location_layer_names} and panel_properties_file: {panel_location_layer_names}"
            )
        # Reorder panel_locations to match panel_properties for easier indexing
        panel_order = [
            panel_properties_layer_names.index(layer)
            for layer in panel_location_layer_names
        ]
        self.panel_locations: list[GeoDataFrame] = [
            panel_locations[i][1] for i in panel_order
        ]

    def validate_panel_properties(
            self, args: GeolocationArgumentParser
    ) -> list[ValidatedGeolocationPanelProperties]:
        panel_properties_filepath = (
            Path(args.panel_properties_file)
            if args.panel_properties_file is not None
            else None
        )
        if panel_properties_filepath is not None:
            if not panel_properties_filepath.exists():
                raise Exception(
                    f"Could not find specified panel properties file: {args.panel_properties_file}"
                )
        else:
            if not (self.dataset / PANEL_PROPERTIES_FILENAME).exists():
                raise Exception(
                    f"No panel properties file found at {self.dataset / PANEL_PROPERTIES_FILENAME}."
                )
        panel_properties_file = self.load_panel_properties(panel_properties_filepath)

        default_properties_names = [
            "default_panel_width",
            "default_panel_height",
            "default_panel_smudge_factor",
            "default_shrink_factor",
        ]
        default_properties = {}
        for prop_name in default_properties_names:
            # Get the default values for the panel properties first from the CLI args
            # and then from the panel_properties file.
            # If one of the defaults is still None
            # and one of the panels does not specify the value the default would be used for an Exception will be thrown
            # in the validate function below
            default_properties[prop_name] = default(
                getattr(args, prop_name), getattr(panel_properties_file, prop_name)
            )
        return validate_geolocation_panel_properties(
            panel_properties_file.panel_properties, default_properties
        )

    def load_panel_properties(
            self, panel_properties_file: Path | None
    ) -> GeolocationPanelPropertiesFile:
        if panel_properties_file is None:
            path = self.dataset / PANEL_PROPERTIES_FILENAME
        else:
            path = panel_properties_file

        return GeolocationPanelPropertiesFile.parse_file(path)

    def convert_orthophotos_to_reflectance(
            self,
            paths: list[Path],
            intensities: NDArray[np.float64],
    ) -> list[list[NDArray[np.float64]] | None]:
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
        with ProgressBar(self.progress, "Converting photos", len(paths)) as pb:
            for photo_index, orthophoto_path in enumerate(paths):
                converted_bands = []
                orthophoto: DatasetReader
                with rasterio.open(orthophoto_path) as orthophoto:
                    photo = orthophoto.read(masked=True, )
                for band_index, band in enumerate(photo):
                    intensities_of_panels = intensities[photo_index, :, band_index]
                    if np.isnan(intensities_of_panels).any():
                        # If for some reason not all intensities are present, we save the indices for debugging purposes
                        # A None value is appended to the converted photos to not lose
                        # the connection between orthophotos and converted photos based on the index
                        unconverted_photos.append((photo_index, band_index))
                        converted_photos.append(None)
                        break
                    coefficients = fit(
                        intensities_of_panels,
                        get_band_reflectance(self.panel_properties, band_index),
                    )
                    converted_bands.append(convert(band, coefficients, band != self.no_data_value))
                else:
                    # For loop did not break, therefore all bands were converted
                    converted_photos.append(converted_bands)
                pb.update()
            if len(unconverted_photos) > 0:
                print("WARNING: Could not convert", len(unconverted_photos), "photos")
            return converted_photos

    def extract_intensities_from_orthophotos(
            self,
            batch_of_orthophotos: list[Path],
            paths_with_visibility: dict[Path, NDArray[np.bool]],
    ) -> NDArray[np.float64]:
        """
        This function extracts intensities from the orthophotos.
        It does so by looking at each photo and determining which panels are visible in the photo.
        If a panel is not visible in the image it is not visible in any of the bands
        as the photos are assumed to rectified and aligned.
        Therefore, extraction is skipped for that panel with `np.Nan` values saved in the output.
        Otherwise, the function uses the recorded gps location of the panel given by `panel_locations`
        to extract the mean intensity of the panel in that band for that photo
        :param paths_with_visibility:
        :param batch_of_orthophotos:
        :return: The extracted intensities for all orthophotos, with np.Nan for values that could not be found.
        """
        task = self.progress.add_task(
            "Extracting intensities", total=len(batch_of_orthophotos), leave=False
        )
        intensities = np.zeros(
            (len(batch_of_orthophotos), len(self.panel_locations), self.number_of_bands)
        )
        for photo_index, orthophoto_path in enumerate(batch_of_orthophotos):
            for panel_index, panel_location in enumerate(self.panel_locations):
                if not paths_with_visibility[orthophoto_path][panel_index]:
                    intensities[photo_index][panel_index] = np.full(
                        self.number_of_bands, np.nan
                    )
                    continue
                # extract the mean intensity for each band at that panel location
                orthophoto: DatasetReader
                with rasterio.open(orthophoto_path) as orthophoto:
                    panel_intensities_per_band = extract_using_geolocation(
                        orthophoto,
                        panel_location,
                        self.panel_properties[panel_index].shrink_factor,
                        self.no_data_value
                    )
                intensities[photo_index][panel_index] = panel_intensities_per_band
            self.progress.update(task, advance=1)
        self.progress.remove_task(task)
        return intensities

    def start(self) -> None:
        all_images_task = self.progress.add_task(
            "Total Progress", total=self.number_of_photos
        )

        paths_with_visibility = {}
        number_of_paths_with_visibility = 0
        with ProgressBar(
                self.progress, "Detecting visible panels", self.number_of_photos
        ) as pb:
            for path in self.orthophoto_paths:
                panels_visible = np.array(
                    [
                        is_panel_in_orthophoto(path, location, self.panel_properties[panel_index].shrink_factor,
                                               self.no_data_value)
                        for panel_index, location in enumerate(self.panel_locations)
                    ]
                )
                number_of_paths_with_visibility += panels_visible.sum() > 0
                paths_with_visibility[path] = panels_visible
                pb.update()

        self.progress.console.print(
            "Number of photos with panels visible: ",
            number_of_paths_with_visibility,
            "/",
            self.number_of_photos,
        ) if self.debug else None

        batches = build_batches_per_full_visibility(paths_with_visibility)

        output_folder = self.dataset / "debug"
        if self.debug:
            for p in (output_folder / "intensity").glob("*.csv"):
                p.unlink()
            for p in (output_folder / "panels").glob("*.tif"):
                p.unlink()
            for path, visibility in paths_with_visibility.items():
                if visibility.sum() == 0:
                    continue
                output_path = get_output_path(
                    self.dataset, path, "panels.tif", "debug/panels"
                )
                run_in_thread(
                    debug_show_geolocation,
                    True,
                    path,
                    self.panel_locations,
                    visibility,
                    [panel.shrink_factor for panel in self.panel_properties],
                    output_path,
                )

        for first_path_is_duplicate, batch in batches:
            # --- Run pipeline
            i = self.extract_intensities_from_orthophotos(batch, paths_with_visibility)
            if self.debug:
                debug_save_intensities(
                    first_path_is_duplicate,
                    i,
                    self.number_of_bands,
                    output_folder / "intensity",
                )
            i = interpolate_intensities(
                i, self.number_of_bands, self.number_of_panels, self.progress
            )
            if self.debug:
                debug_save_intensities(
                    first_path_is_duplicate,
                    i,
                    self.number_of_bands,
                    output_folder / "intensity",
                    "_interpolated",
                )
            c = self.convert_orthophotos_to_reflectance(batch, i)
            del i
            save_orthophotos(self.dataset, batch, c, self.progress)
            del c
            self.progress.update(all_images_task, advance=len(batch))
        if self.debug:
            debug_combine_and_plot_intensities(
                self.number_of_photos,
                self.number_of_bands,
                self.number_of_panels,
                output_folder / "intensity",
                self.debug_dpi
            )


def main() -> None:
    # --- Get input arguments from user
    args = GeolocationArgumentParser(
        formatter_class=RichHelpFormatter,
        description="Automatically detect reflection calibration panels in images and transform the given images to "
                    "reflectance",
        epilog="If you have any questions, please contact us via the github repository issues",
    ).parse_args()
    if not is_tool_installed("exiftool"):
        raise Exception("Exiftool is not installed. Follow the readme to install it")

    install(show_locals=args.debug, suppress=[reflectdetect] if not args.debug else [])
    GeolocationEngine(args).start()


if __name__ == "__main__":
    main()
