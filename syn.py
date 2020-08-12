import sys
sys.path.insert(0, ".")
import matplotlib.pyplot as plt
import quaternion, os, argparse, glob
import numpy as np
from tqdm import tqdm
os.environ["DEBUG"] = ""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils

from models.z_buffermodel import ZbufferModelPts
from options.options import get_model

torch.backends.cudnn.enabled = True

# synsin is z buffer model
MODEL_PATH = ""
BATCH_SIZE = 4
device = 'cuda'
DIR = args.indir

class SynsinTransformer(object):
  def __init__(self, model_path):
    super().__init__()
    self.model_path = model_path

    # the option is stored in the pth file
    opts = torch.load(MODEL_PATH)['opts']
    opts.render_ids = [1]

    self.model = ZbufferModelPts(opts).cuda()
    self.model.load_state_dict(torch.load(MODEL_PATH)['state_dict'], strict=False)
    self.model.eval()
    print(self.model)

    self.K, self.Kinv = get_K()
  
  def __call__(self, x, K=None, Kinv=None, DIST=None, step_num=None):
    # handling default argument
    if DIST is None:
      DIST = [0.04, 0.04, 0.04, 0.04, 0.08]
      SN = 4
    if K is None:
      K = self.K
      Kinv = self.Kinv
    
    param = get_param(DIST, step_num)
    RTs = param2RT(param)

    pred_imgs = []
    masks = []

    # return the transformations for each image
    for img in x:
      batch = {
        'images' : [x.unsqueeze(0)],
        'cameras' : [{'K' : K, 'Kinv' : Kinv}]
      }

      # Get parameters for the transformation

      # Generate a new view at the new transformation
      with torch.no_grad():
        # list of (1, 3, 256, 256) [-1, 1]
        img, mask = self.model.forward_angle(batch, RTs, get_mask=True)
        masks.append(mask)
        pred_imgs.append(img)

    return pred_imgs, masks


def get_RT(theta, phi, tx, ty, tz):
  RT = torch.eye(4).unsqueeze(0)
  # Set up rotation
  RT[0,0:3,0:3] = torch.Tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_vector([phi, theta, 0])))
  # Set up translation
  RT[0,0:3,3] = torch.Tensor([tx, ty, tz])
  return RT


def get_K():
  K = np.array([
    [915.39183868 / 256, 0.0, 482.90628282 / 256, 0],
    [0.0, 925.59365333 / 256, 228.69102112 / 256, 0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]])
  Kinv = np.linalg.inv(K).astype(np.float32)
  K = torch.from_numpy(K).unsqueeze(0).float()
  Kinv = torch.from_numpy(Kinv).unsqueeze(0).float()
  return K, Kinv


def get_param(DIST, SN=4):
  return [
    list(np.linspace(0, DIST[i], SN)) + \
    list(np.linspace(0, -DIST[i], SN))
    for i in range(5)]


def param2RT(param):
  RTs = []
  for i, param in enumerate(params):
    RTs.extend([[p if j == i else 0 for j in range(5)]
      for p in param])
  RTs = [get_RT(*p) for p in RTs]

