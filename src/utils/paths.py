import os


def get_extraction_path(image_path: str) -> (str, str):
    image_filename = image_path.split("/")[-1]
    path = "/".join(image_path.split("/")[:-1])
    return path + "/metadata/", image_filename.split(".")[0] + "_extraction.json"

def get_detection_path(image_path: str) -> (str, str):
    image_filename = image_path.split("/")[-1]
    path = "/".join(image_path.split("/")[:-1])
    return path + "/metadata/", image_filename.split(".")[0] + "_detection.json"

def get_output_path(path, new_ending, folder):
    filename = path.as_posix().split("/")[-1].split(".")[
                   0] + "_" + new_ending + ".tif"
    output_folder = "/".join(path.as_posix().split("/")[:-1]) + "/" + folder + "/"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder + filename



def get_extractions_path(image_path: str) -> (str, str):
    path = "/".join(image_path.split("/")[:-1])
    return path + "/metadata/", str(get_image_id(image_path)) + "_extraction.json"


def get_transformed_path(image_path: str) -> (str, str):
    image_filename = image_path.split("/")[-1]
    path = "/".join(image_path.split("/")[:-1])
    return path + "/reflectance/", image_filename.split(".")[0] + "_transformed.png"


def get_image_band(image_path: str) -> int:
    # Assumes filenames are *_{band}.* with {band} being the 1 indexed band number
    return int(image_path.split("/")[-1].split(".")[0].split("_")[-1])


def get_image_id(image_path: str) -> int:
    # Assumes filenames are *_{id}_band.*
    return int(image_path.split("_")[-2])
