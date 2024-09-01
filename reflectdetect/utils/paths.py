import os
from pathlib import Path


def get_output_path(path: Path, new_ending: str, folder: str) -> Path:
    filename = path.as_posix().split("/")[-1].split(".")[
                   0] + "_" + new_ending + ".tif"
    output_folder = "/".join(path.as_posix().split("/")[:-2]) + "/" + folder + "/"
    os.makedirs(output_folder, exist_ok=True)
    return Path(output_folder + filename)
