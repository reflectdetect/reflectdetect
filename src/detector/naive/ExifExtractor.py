import exifread
import json


def get_exif_exist(data, key):
    return data.get(key, None)

def convert_to_meters(value):
    # Convert the IfdTag value to a float assuming it's a Ratio
    if isinstance(value.values[0], exifread.utils.Ratio):
        ratio = value.values[0]
        return float(ratio.num) / float(ratio.den)
    return None

def get_exif_altitude(exif_data):
    alt = None
    focal_length_m = None
    resolution = None

    gps_altitude = get_exif_exist(exif_data, 'GPS GPSAltitude')
    gps_altitude_ref = get_exif_exist(exif_data, 'GPS GPSAltitudeRef')
    focal_length = get_exif_exist(exif_data, 'EXIF FocalLength')
    image_width = get_exif_exist(exif_data, 'Image ImageWidth')
    image_length = get_exif_exist(exif_data, 'Image ImageLength')

    if gps_altitude and gps_altitude_ref:
        alt = convert_to_meters(gps_altitude) / 1000 # convert from cm to meter
        if gps_altitude_ref.values[0] == 1:
            alt = -alt

    if focal_length:
        focal_length_m = convert_to_meters(focal_length)

    if image_width and image_length:
        resolution = [int(image_width.values[0]), int(image_length.values[0])]

    return {"altitude": alt, "focal_length_m": focal_length_m, "resolution": resolution}

def write_exif_data_to_json(image_path, output_json_path):
    # Open image file for reading (binary mode)
    with open(image_path, 'rb') as f:
        # Return Exif tags
        tags = exifread.process_file(f)

    # Get the desired exif data
    exif_data = get_exif_altitude(tags)

    # Write to JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(exif_data, json_file, indent=4)
