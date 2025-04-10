"""
author: Zooey He
email: zhuohonghe@gmail.com
date: 2025-04-09
"""

import cv2
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
