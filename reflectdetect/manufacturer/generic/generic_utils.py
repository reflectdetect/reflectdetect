from rich.prompt import IntPrompt


def raw_image_to_radiance(raw_image, bits_per_pixel: int):
    bit_depth_max = float(2 ** bits_per_pixel)
    radiance_image = raw_image.astype(float) / bit_depth_max
    return radiance_image
