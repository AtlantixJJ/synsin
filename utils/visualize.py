import numpy as np
from matplotlib import cm
import cv2


POSITIVE_COLOR = cm.get_cmap("Reds")
NEGATIVE_COLOR = cm.get_cmap("Blues")
def heatmap_numpy(image):
    """
    assume numpy array as input: (N, H, W) in [0, 1]
    returns: (N, H, W, 3)
    """
    image1 = image.copy()
    mask1 = image1 > 0
    image1[~mask1] = 0

    image2 = -image.copy()
    mask2 = image2 > 0
    image2[~mask2] = 0

    pos_img = POSITIVE_COLOR(image1)[:, :, :, :3]
    neg_img = NEGATIVE_COLOR(image2)[:, :, :, :3]

    x = np.ones_like(pos_img)
    x[mask1] = pos_img[mask1]
    x[mask2] = neg_img[mask2]

    return x


class VideoWriter(object):
  """Defines the video writer.

  This class can be used to create a video.

  NOTE: `.avi` and `DIVX` is the most recommended codec format since it does not
  rely on other dependencies.
  """

  def __init__(self, path, frame_height, frame_width, fps=24, codec='DIVX'):
    """Creates the video writer."""
    self.path = path
    self.frame_height = frame_height
    self.frame_width = frame_width
    self.fps = fps
    self.codec = codec

    self.video = cv2.VideoWriter(filename=path,
                                 fourcc=cv2.VideoWriter_fourcc(*codec),
                                 fps=fps,
                                 frameSize=(frame_width, frame_height))

  def __del__(self):
    """Releases the opened video."""
    self.video.release()

  def write(self, frame):
    """Writes a target frame.

    NOTE: The input frame is assumed to be with `RGB` channel order.
    """
    self.video.write(frame[:, :, ::-1])