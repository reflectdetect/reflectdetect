from exiftool import ExifToolHelper


def get_camera_properties(image_path):
    with ExifToolHelper() as et:
        metadata = et.get_metadata(image_path)[0]
        # focal_length_mm, focal_plane_x_res, focal_plane_y_res, focal_plane_resolution_unit
    return metadata["EXIF:FocalLength"], metadata["EXIF:FocalPlaneXResolution"], metadata["EXIF:FocalPlaneYResolution"], \
        metadata[
            "EXIF:FocalPlaneResolutionUnit"],