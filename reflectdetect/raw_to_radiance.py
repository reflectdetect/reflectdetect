from pathlib import Path

from exiftool import ExifToolHelper
from matplotlib import pyplot as plt
from rich.pretty import install
from rich.progress import track
from rich.prompt import IntPrompt
from rich_argparse import RichHelpFormatter
from tap import Tap

from reflectdetect.constants import CONVERTED_FILE_ENDING
from reflectdetect.manufacturer.generic import generic_utils
from reflectdetect.manufacturer.micasense import micasense_utils
from reflectdetect.manufacturer.micasense.micasense_metadata import Metadata
from reflectdetect.utils.apriltags import save_images
from reflectdetect.utils.paths import is_tool_installed

supported_manufacturers = ["micasense", "generic"]

class ConverterArgumentParser(Tap):
    manufacturer: str = "generic"
    dataset: Path  # Path to the dataset folder

    def configure(self) -> None:
        self.add_argument("dataset", nargs="?", default=".", type=Path)
        self.add_argument("--manufacturer", choices=supported_manufacturers)


def main() -> None:
    install()
    args = ConverterArgumentParser(
        formatter_class=RichHelpFormatter,
        description="Converts raw image to radiance to be then used with the reflectdetect tool",
        epilog="If you have any questions, please contact us via the github repository issues",
    ).parse_args()

    if not is_tool_installed("exiftool"):
        raise Exception("Exiftool is not installed. Follow the readme to install it")
    et = ExifToolHelper(True, True, True)
    if not Path(args.dataset).exists():
        raise Exception("Dataset path does not exists")

    if not (Path(args.dataset) / "raw").exists():
        raise Exception("In the dataset folder there should be a folder called 'raw' for the image files")
    paths = sorted(list((Path(args.dataset) / "raw").glob("*.tif")))

    print("Available manufactuers:", supported_manufacturers)
    if args.manufacturer not in supported_manufacturers:
        raise Exception(f"Manufacturer no supported: {args.manufacturer} not in {supported_manufacturers}")
    print("Converting for manufacturer:", args.manufacturer)
    bits_per_pixel = None
    if args.manufacturer == "generic":
        bits_per_pixel = IntPrompt.ask("Enter the bitdepth of the raw images",
                                       default=16)
    for path in track(paths, description="Converting raw images to radiance"):
        converted_image = None
        image_raw = plt.imread(path)

        if args.manufacturer == "micasense":
            meta = Metadata(path.as_posix())
            image_rad, _, _, _ = micasense_utils.raw_image_to_radiance(meta, image_raw)
            image_rad_undistorted = micasense_utils.correct_lens_distortion(meta, image_rad)  # type: ignore
            converted_image = image_rad_undistorted
        if args.manufacturer == "generic":
            converted_image = generic_utils.raw_image_to_radiance(image_raw, bits_per_pixel)
        if converted_image is None:
            raise Exception("Could not convert image")
        save_images(et, args.dataset, [path], [converted_image], None, "images", CONVERTED_FILE_ENDING)


if __name__ == '__main__':
    main()
