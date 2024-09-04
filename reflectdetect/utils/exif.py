from pathlib import Path
from typing import Any

from exiftool import ExifToolHelper

from reflectdetect.utils.thread import run_in_thread


def get_camera_properties(image_path: Path) -> tuple[Any, Any, Any, Any]:
    """
    Extracts the camera properties from the exif data of a given image using exiftool
    :param image_path: path to the image
    :return: the exif data of the image:
     focal_length_mm, focal_plane_x_res, focal_plane_y_res, focal_plane_resolution_unit
    """
    with ExifToolHelper() as et:
        metadata = run_in_thread(et.get_metadata, True, image_path.as_posix())[0]  # type: ignore
        # focal_length_mm, focal_plane_x_res, focal_plane_y_res, focal_plane_resolution_unit
    return (
        metadata["EXIF:FocalLength"],
        metadata["EXIF:FocalPlaneXResolution"],
        metadata["EXIF:FocalPlaneYResolution"],
        metadata["EXIF:FocalPlaneResolutionUnit"],
    )
