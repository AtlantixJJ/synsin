import numpy as np

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
