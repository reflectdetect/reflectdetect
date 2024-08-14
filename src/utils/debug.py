import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from robotpy_apriltag import AprilTagDetection


def show_panel(img, tags: list[AprilTagDetection], corners: list[float], output_path: str | None = None):
    fig_2d = plt.figure()
    ax = fig_2d.subplots(1, 1)
    ax.imshow(img, cmap="grey")

    for tag in tags:
        ax.scatter(tag.getCenter().x, tag.getCenter().y)
    x, y = zip(*corners)

    # Append the first point to the end to close the rectangle/polygon
    x = list(x) + [x[0]]
    y = list(y) + [y[0]]
    ax.plot(x, y, linewidth=1)

    if output_path is not None:
        fig_2d.savefig(output_path)
    else:
        fig_2d.show()
    plt.close(fig_2d)


def show_intensities(intensities, output_path: str | None = None):
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


def debug_combine_and_plot_intensities(number_of_images, number_of_bands, output_folder, panel_properties, suffix=""):
    intensities = np.zeros((number_of_images, len(panel_properties), number_of_bands))
    for band in range(0, number_of_bands):
        output_path = Path(
            output_folder + "/band_" + str(band) + "_intensities" + suffix + ".csv")
        intensities[:, :, band] = np.genfromtxt(output_path, delimiter=",")
    output_path = Path(
        output_folder + "/intensities" + suffix + ".tif")
    show_intensities(intensities, output_path.as_posix())


def debug_save_intensities(i, number_of_bands, output_folder, suffix=""):
    for band in range(0, number_of_bands):
        output_path = Path(output_folder + "/band_" + str(band) + "_intensities" + suffix + ".csv")
        with open(output_path, "a") as f:
            f.write("\n")
            data = i[:, :, band].astype(str)
            data[data == 'nan'] = ''
            np.savetxt(f, data, delimiter=",", fmt="%s")


def debug_save_intensities_single_band(i, band_index, output_folder, suffix=""):
    output_path = Path(output_folder + "/band_" + str(band_index) + "_intensities" + suffix + ".csv")
    with open(output_path, "a") as f:
        f.write("\n")
        data = i[:, :].astype(str)
        data[data == 'nan'] = ''
        np.savetxt(f, data, delimiter=",", fmt="%s")
