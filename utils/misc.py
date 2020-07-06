# python 3.7
"""Misc utility functions."""

import os
import numpy as np
from PIL import Image


def imread(fpath):
  with open(os.path.join(fpath), "rb") as f:
    return np.asarray(Image.open(f), dtype="uint8")


def imread_pil(fpath):
  with open(os.path.join(fpath), "rb") as f:
    return Image.open(f)


def imwrite(fpath, image, format="RGB"):
  """
  image: np array, value range in [0, 255].
  """
  if ".jpg" in fpath or ".jpeg" in fpath:
    ext = "JPEG"
  elif ".png" in fpath:
    ext = "PNG"
  with open(os.path.join(fpath), "wb") as f:
    Image.fromarray(image.astype("uint8")).convert(format).save(f, format=ext)
