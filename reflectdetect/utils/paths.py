import os
from pathlib import Path


def get_filename(path: Path) -> str:
    return path.as_posix().split("/")[-1].split(".")[0]


def get_output_path(dataset: Path, filepath: Path, new_ending: str, folder: str) -> Path:
    filename = get_filename(filepath) + "_" + new_ending
    output_folder = dataset / folder
    os.makedirs(output_folder, exist_ok=True)
    return output_folder / filename
