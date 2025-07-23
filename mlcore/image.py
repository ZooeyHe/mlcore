"""
author: Zooey He
email: zhuohonghe@gmail.com
date: 2025-04-09
"""

import cv2
import os
import torch
from PIL import Image
import numpy as np
from einops import rearrange

from mlcore.file import get_file_type, file_exists

"""
Utilities related to images.
"""

depth_est_pipe = None

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
        specs. The channels are R,G,B.
    """
    if not file_exists(filepath) or not get_file_type(filepath).startswith("image"):
        raise ValueError(f"{filepath} is not a valid image file.")
    
    img_arr = cv2.imread(filepath)  # Reads the image as BGR
    img_arr = img_arr[:,:,[2,1,0]]  # Convert BGR to RGB

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


def grayscale(
        img: np.ndarray,
        mode: int = cv2.COLOR_RGB2GRAY,
        keepdims: bool = False
    ) -> np.ndarray:
    """
    Grayscale an image (H W 3) to (H W 1) if keepdims is false and (H W 3) if keepdims is true.
    """
    img = cv2.cvtColor(img, mode)
    if keepdims:
        img = np.repeat(img, 3, 2)
    return img

def compute_disparity(
        img_1: np.ndarray,
        img_2: np.ndarray,
        numDisparities=16,
        blockSize=15
    ) -> np.ndarray:
    """
    Compute the disparity between two rectified images.

    Args:
        img_1: image 1, SHAPE H W C
        img_2: image 2, SHAPE H W C
        numDisparities: number of disparities.
        blockSize: The block size

    Returns:
        disparity: a disparity map using uint8 (H W)
    """
    # Grayscale.
    if img_1.shape[2] == 3:
        img_1 = grayscale(img_1)
    if img_2.shape[2] == 3:
        img_2 = grayscale(img_2)

    # Initialize the stereo block matching object 
    stereo = cv2.StereoBM.create(numDisparities=numDisparities, blockSize=blockSize)

    # Compute the disparity image
    disparity = stereo.compute(img_1, img_2)

    # Normalize the image for representation
    min = disparity.min()
    max = disparity.max()
    disparity = np.uint8(6400 * (disparity - min) / (max - min))

    return disparity


class DepthEstimator():
    """
    Singleton depth estimator. TODO make this thread safe
    """
    _instance = None
    _pipeline = None

    def __new__(cls, *args, **kwargs):
        try:
            from transformers import pipeline
        except:
            raise ImportError("Unable to import hf transformers. Please `pip install transformers`")
        
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._pipeline = pipeline(
                task="depth-estimation", model=kwargs["model"], use_fast=True, device=kwargs["device"]
            )
        return cls._instance

    def __init__(self, model: str = "depth-anything/Depth-Anything-V2-Small-hf", device: int = 0):
        pass

    def __call__(self, img: np.ndarray, batch_size: int = None):
        """
        Estimate depth using a transformers pipeline.

        Args:
            img: the rgb image data (uint8, [0-255]). Shape: N H W C || H W C
            batch_size: the batch size to use.

        Return:
            depth: the estimated depth (uint8, [0-255]). Shape: N H W || H W
        """
        if isinstance(img, np.ndarray):
            if len(img.shape) == 4:
                img = [im for im in img]
            else:
                img = [img]
        elif isinstance(img, list):
            pass
        else:
            raise ValueError("Invalid img argument type")
        
        outputs = self._pipeline([Image.fromarray(im) for im in img], batch_size=batch_size)

        if len(img) == 1:
            return outputs[0]["depth"]
        else:
            return np.stack([d["depth"] for d in outputs], axis=0)


def estimate_depth(
    img: np.ndarray, 
    model_hf: str = "depth-anything/Depth-Anything-V2-Small-hf",
    batch_size: int = None,
    device: int = 0
):
    """
    Estimate the depth. This method is deprecated wrt to the DepthEstimator singleton class.
    """
    try:
        from transformers import pipeline
        # from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except:
        raise ImportError("Unable to import hf transformers. Please `pip install transformers`")

    if isinstance(img, np.ndarray):
        if len(img.shape) == 4:
            img = [im for im in img]
        else:
            img = [img]
    elif isinstance(img, list):
        pass
    else:
        raise ValueError("Invalid img argument type")

    global depth_est_pipe

    if depth_est_pipe is None:
        depth_est_pipe = pipeline(task="depth-estimation", model=model_hf, use_fast=True, device=device)
    
    outputs = depth_est_pipe([Image.fromarray(im) for im in img], batch_size=batch_size)

    if len(img) == 1:
        return outputs[0]["depth"]
    else:
        return np.stack([d["depth"] for d in outputs], axis=0)
