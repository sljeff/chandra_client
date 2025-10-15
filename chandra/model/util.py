import math
from typing import Tuple

from PIL import Image


def scale_to_fit(
    img: Image.Image,
    max_size: Tuple[int, int] = (3072, 2048),
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


def detect_repeat_token(
    predicted_tokens: str, max_repeats: int = 4, window_size: int = 50
):
    if len(predicted_tokens) < window_size:
        return False

    # Look at the last window_size tokens
    recent_tokens = predicted_tokens[-window_size:].lower()

    # Try different sequence lengths (1 to window_size//2)
    for seq_len in range(1, window_size // 2 + 1):
        # Skip if we can't fit enough repetitions
        if seq_len * (max_repeats + 1) > window_size:
            continue

        # Extract the potential repeating sequence from the end
        candidate_seq = recent_tokens[-seq_len:]

        # Count how many times this sequence appears consecutively at the end
        repeat_count = 0
        pos = len(recent_tokens) - seq_len

        while pos >= 0:
            if recent_tokens[pos : pos + seq_len] == candidate_seq:
                repeat_count += 1
                pos -= seq_len
            else:
                break

        # If we found more than max_repeats consecutive occurrences
        if repeat_count > max_repeats:
            return True

    return False
