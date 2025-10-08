import math
from typing import Tuple

from PIL import Image


def scale_to_fit(
    img: Image.Image,
    max_size: Tuple[int, int] = (1024, 1024),
    min_size: Tuple[int, int] = (28, 28),
):
    resample_method = Image.Resampling.LANCZOS

    width, height = img.size

    # Check for empty or invalid image
    if width == 0 or height == 0:
        return img

    max_width, max_height = max_size
    min_width, min_height = min_size

    current_pixels = width * height
    max_pixels = max_width * max_height
    min_pixels = min_width * min_height

    if current_pixels > max_pixels:
        scale_factor = (max_pixels / current_pixels) ** 0.5

        new_width = math.floor(width * scale_factor)
        new_height = math.floor(height * scale_factor)
    elif current_pixels < min_pixels:
        scale_factor = (min_pixels / current_pixels) ** 0.5

        new_width = math.ceil(width * scale_factor)
        new_height = math.ceil(height * scale_factor)
    else:
        return img

    return img.resize((new_width, new_height), resample=resample_method)