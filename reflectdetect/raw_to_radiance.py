from pathlib import Path

from exiftool import ExifToolHelper
from matplotlib import pyplot as plt
from rich.progress import track
from rich_argparse import RichHelpFormatter
from tap import Tap

from reflectdetect.manufacturer_utils import micasense_utils
from reflectdetect.utils.apriltags import save_images
from reflectdetect.utils.paths import is_tool_installed


class ConverterArgumentParser(Tap):
    manufacturer: str = "generic"
    dataset: Path  # Path to the dataset folder

    def configure(self) -> None:
        self.add_argument("dataset", nargs="?", default=".", type=Path)
        self.add_argument("-d", "--debug")


if __name__ == '__main__':
    args = ConverterArgumentParser(
        formatter_class=RichHelpFormatter,
        description="Converts raw image to radiance to be then used with the reflectdetect tool",
        epilog="If you have any questions, please contact us via the github repository issues",
    ).parse_args()

    if not is_tool_installed("exiftool"):
        raise Exception("Exiftool is not installed. Follow the readme to install it")

    if not Path(args.dataset).exists():
        raise Exception("Dataset path does not exists")

    if not (Path(args.dataset) / "raw").exists():
        raise Exception("In the dataset folder there should be a folder called 'raw' for the image files")
    paths = sorted(list((Path(args.dataset) / "raw").glob("*.tif")))
    converted_images = []
    for path in track(paths, description="Converting raw images to radiance"):
        image_raw = plt.imread(path)
        with ExifToolHelper() as et:
            metadata = run_in_thread(et.get_metadata, True, image_path.as_posix())[0]  # type: ignore

        if args.manufacturer == "micasense":
            image_rad = micasense_utils.raw_image_to_radiance(metadata, image_raw)
            image_rad_undistorted = micasense_utils.correct_lens_distortion(metadata, image_rad)
            converted_images.append(image_rad_undistorted)
        if args.manufacturer == "generic":
            pass
    save_images(args.dataset, paths, converted_images, None, "images", ".tif")
