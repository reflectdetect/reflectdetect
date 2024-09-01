import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import rasterio
import shapely
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from rasterio import DatasetReader
from rasterio.plot import show
from rich.progress import Progress, TaskID
from robotpy_apriltag import AprilTagDetection
from shapely import Polygon

from reflectdetect.utils.polygons import shrink_or_swell_shapely_polygon
from reflectdetect.utils.thread import run_in_thread

matplotlib.use('Agg')


def debug_show_geolocation(img: DatasetReader, panels_corners: list[Polygon], shrink_factor: float,
                           output_path: Path | None = None) -> None:
    fig_2d = plt.figure()
    ax = fig_2d.subplots(1, 1)
    ax.axis("off")
    rasterio.plot.show(img, ax=ax)

    for corners in panels_corners:
        x, y = corners.exterior.xy

        # Append the first point to the end to close the rectangle/polygon
        x = list(x) + [x[0]]
        y = list(y) + [y[0]]
        ax.plot(x, y, linewidth=1)
        polygon = shrink_or_swell_shapely_polygon(corners, shrink_factor)
        detection_corners = polygon.exterior.coords.xy
        x, y = detection_corners
        x = list(x) + [x[0]]
        y = list(y) + [y[0]]
        ax.plot(x, y, linewidth=1, linestyle="dotted")

    if output_path is not None:
        fig_2d.savefig(output_path, dpi=300)
    else:
        fig_2d.show()
    plt.close(fig_2d)


def debug_show_panel(img: NDArray[np.float64], tag: AprilTagDetection, corners: list[float],
                     shrink_factor: float,
                     output_path: Path | None = None) -> None:
    fig_2d = plt.figure()
    ax = fig_2d.subplots(1, 1)
    ax.axis("off")
    ax.imshow(img, cmap="grey")

    ax.scatter(tag.getCenter().x, tag.getCenter().y)
    x, y = zip(*corners)

    # Append the first point to the end to close the rectangle/polygon
    x = list(x) + [x[0]]
    y = list(y) + [y[0]]
    ax.plot(x, y, linewidth=1)
    polygon = shapely.Polygon(corners)
    polygon = shrink_or_swell_shapely_polygon(polygon, shrink_factor)
    detection_corners = polygon.exterior.coords.xy
    x, y = detection_corners
    x = list(x) + [x[0]]
    y = list(y) + [y[0]]
    ax.plot(x, y, linewidth=1, linestyle="dotted")

    if output_path is not None:
        fig_2d.savefig(output_path)
    else:
        fig_2d.show()
    plt.close(fig_2d)


def show_intensities(intensities: NDArray[np.float64], output_path: str | None = None) -> None:
    fig, axes = plt.subplots(len(intensities[0, 0, :]), sharex=True, figsize=(15, 15))
    max_intensity = np.nanmax(intensities)

    for band_index, ax in enumerate(axes):
        ax.yaxis.set_ticks([])
        ax.set_ylim([0, max_intensity * 1.2])
        ax.set_ylabel("Intensity")
        ax.annotate(
            "Band " + str(band_index),
            xy=(0, 1), xycoords='axes fraction',
            xytext=(+0.5, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
        for panel_index in range(0, len(intensities[0, :, 0])):
            ax.plot(intensities[:, panel_index, band_index])
    plt.xlabel("Image index")
    plt.xlim([0, len(intensities[:, 0, 0])])
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close(fig)


def debug_combine_and_plot_intensities(number_of_images: int,
                                       number_of_bands: int,
                                       number_of_panels: int,
                                       output_folder: Path,
                                       suffix: str = "") -> None:
    intensities = np.zeros((number_of_images, number_of_panels, number_of_bands))
    for band in range(0, number_of_bands):
        filename = f"band_{band}_intensities{suffix}.csv"
        output_path = output_folder / filename
        intensities[:, :, band] = np.genfromtxt(output_path, delimiter=",")
    output_path = output_folder / f"intensities{suffix}.tif"
    run_in_thread(show_intensities, intensities, output_path.as_posix())


def debug_save_intensities(first_path_is_duplicate: bool, intensities: NDArray[np.float64], number_of_bands: int,
                           output_folder: Path,
                           suffix: str = "") -> None:
    os.makedirs(output_folder, exist_ok=True)
    for band in range(0, number_of_bands):
        filename = f"band_{band}_intensities{suffix}.csv"
        output_path = output_folder / filename
        with open(output_path, "a") as f:
            f.write("\n")
            data = intensities[1:, :, band].astype(str) if first_path_is_duplicate else intensities[:, :, band].astype(
                str)
            data[data == 'nan'] = ''
            np.savetxt(f, data, delimiter=",", fmt="%s")


def debug_save_intensities_single_band(intensities: NDArray[np.float64], band_index: int, output_folder: Path,
                                       suffix: str = "") -> None:
    os.makedirs(output_folder, exist_ok=True)
    output_path = output_folder / f"band_{str(band_index)}_intensities{suffix}.csv"
    with open(output_path, "a+") as f:
        f.write("\n")
        data = intensities[:, :].astype(str)
        data[data == 'nan'] = ''
        np.savetxt(f, data, delimiter=",", fmt="%s")


class ProgressBar:
    def __init__(self, progress: Progress | None, description: str, total: int | None = None) -> None:
        self.progress = progress
        self.description = description
        self.total = total
        self.task: TaskID | None = None

    def __enter__(self) -> 'ProgressBar':
        if self.progress is not None:
            self.task = self.progress.add_task(self.description, total=self.total, leave=False)
        return self

    def __exit__(self, exc_type: type | None, exc_value: Exception | None,
                 traceback: Any | None) -> None:
        if self.progress is not None and self.task is not None:
            self.progress.remove_task(self.task)

    def update(self, advance: int = 1) -> None:
        if self.progress is not None and self.task is not None:
            self.progress.update(self.task, advance=advance)
