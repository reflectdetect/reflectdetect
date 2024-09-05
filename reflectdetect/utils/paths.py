import os
from pathlib import Path
from typing import Any


def get_filename(path: Path) -> str:
    """
    Get only the filename without the file ending from a path
    :param path: The path the filename will be extracted from
    :return: The extracted filename
    """
    return path.as_posix().split("/")[-1].split(".")[0]


def get_output_path(
    dataset: Path, filepath: Path, new_ending: str, folder: str
) -> Path:
    """
    Gets the output path of a new file based on a filepath.
    It extracts the filename form the filepath and saves it with new ending to the specified folder in the dataset

    :param dataset: The main folder
    :param filepath: filepath the filename will be based on
    :param new_ending: new file ending
    :param folder: subfolder of the dataset to but the new file into
    :return: a path of the new file
    """
    filename = get_filename(filepath) + "_" + new_ending
    output_folder = dataset / folder
    os.makedirs(output_folder, exist_ok=True)
    return output_folder / filename


def default(value: Any, default: Any) -> Any:
    """
    :param value: some value
    :param default: some default
    :return:  the value if it's not none else the default
    """
    return value if value is not None else default


def is_tool_installed(name: str) -> bool:
    # https://stackoverflow.com/a/34177358
    """
    :param name: name of the executable tool
    :return: whether it is on PATH and marked as executable
    """

    from shutil import which

    return which(name) is not None
