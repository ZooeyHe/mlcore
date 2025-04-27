"""
author: Zooey He
email: zhuohonghe@gmail.com
date: 2025-04-09
"""

import cv2
import os
from PIL import Image
import numpy as np
from einops import rearrange

from mlcore.file import get_file_type, file_exists

"""
Utilities related to images.
"""


# Image loading from disk
def read_img(filepath: str, out_ein: str = None) -> np.ndarray:
    """
    Given an image at a filepath return the image data.

    Args:
        filepath: the path to an image to load.
        out_ein: the einstien notation which describes the output. For example, "H W C" or "W H C".
            If None given, returns the default shape of the image (H W C). 

    Returns:
        img_arr: The image array in (H W C, uint8) by default, but will be reformatted to out_ein
        specs.
    """
    if not file_exists(filepath) or not get_file_type(filepath).startswith("image"):
        raise ValueError(f"{filepath} is not a valid image file.")
    
    img_arr = cv2.imread(filepath)

    if out_ein:
        return rearrange(img_arr, f"H W C -> {out_ein}")
    else:
        return img_arr


def save_img(img_arr: np.ndarray, filepath: str, img_ein: str = None) -> None:
    """
    Given an image array, save it to file.

    Args:
        img_arr: a np.ndarray, a numpy array which contains image data.
        filepath: a save filepath (.png).
        img_ein: the np.ndarray image's ein. Default is "H W C"
    """
    if not os.path.splitext(filepath)[1] in [".png", ".jpg", ".jpeg"]:
        raise ValueError("Expecting image filepath to be .png")
    
    if img_ein:
        img_arr = rearrange(img_arr, f"{img_ein} -> H W C")

    if img_arr.dtype == np.uint8 and img_arr.max() < 256 and img_arr.min() >= 0:
        pass  # Already acceptable type.
    else:
        # Convert the array values to between 0 - 255
        img_arr = img_arr.astype(np.float64)
        img_arr = img_arr / img_arr.max()
        img_arr = (img_arr * 255).astype(np.uint8)

    img = Image.fromarray(img_arr)

    img.save(filepath)

    