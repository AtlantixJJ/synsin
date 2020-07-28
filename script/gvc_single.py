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

from models.networks.sync_batchnorm import convert_model
from models.projection.z_buffer_manipulator import PtsManipulator
from models.base_model import BaseModel
from data.simple import SimpleDataset
from utils.misc import imread_pil, imwrite, torch2numpy
from utils.visualize import heatmap_numpy, VideoWriter
from options.options import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--model-path',
  default='./modelcheckpoints/realestate/synsin.pth')
parser.add_argument('--MPI',
  default='0,1')
parser.add_argument('--indir',
  default='mydata/stylegan_bedroom_synthesis')
parser.add_argument('--outdir',
  default='mydata/stylegan_bedroom_transform')
args = parser.parse_args()
torch.backends.cudnn.enabled = True

if "," not in args.MPI:
  world_size = int(args.MPI)
  gpus = [0, 1, 2, 3, 4, 5, 6, 7]
  for i in range(world_size):
    os.system(f"CUDA_VISIBLE_DEVICES={gpus[i]} python script/gvc_single.py --model-path {args.model_path} --indir {args.indir} --outdir {args.outdir} --MPI {i},{world_size} &")
  exit(0)

rank, world_size = args.MPI.split(",")
rank = int(rank)
world_size = int(world_size)

# synsin is z buffer model
MODEL_PATH = args.model_path
BATCH_SIZE = 4
device = 'cuda'
DIR = args.indir

# the option is stored in the pth file
opts = torch.load(MODEL_PATH)['opts']
opts.render_ids = [1]

model = get_model(opts)

if 'sync' in opts.norm_G:
  model = convert_model(model)
model = nn.DataParallel(model).cuda()

model_to_test = BaseModel(model, opts)
model_to_test.load_state_dict(torch.load(MODEL_PATH)['state_dict'], strict=False)
model_to_test.eval()

transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop((256, 256)),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def get_RT(theta, phi, tx, ty, tz):
  RT = torch.eye(4).unsqueeze(0)
  # Set up rotation
  RT[0,0:3,0:3] = torch.Tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_vector([phi, theta, 0])))
  # Set up translation
  RT[0,0:3,3] = torch.Tensor([tx, ty, tz])
  return RT

def circle(s, n):
  x = list(np.linspace(0, -s, SN))
  x = x + list(np.linspace(-s, s, 2 * SN))
  x = x + list(np.linspace(s, 0, SN))
  return x


K = np.array([
  [915.39183868 / 256, 0.0, 482.90628282 / 256, 0],
  [0.0, 925.59365333 / 256, 228.69102112 / 256, 0],
  [0.0, 0.0, 1.0, 0.0],
  [0.0, 0.0, 0.0, 1.0]])

"""
offset = np.array(
    [[2, 0, -1, 0], [0, -2, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]],# Flip ys to match habitat
    dtype=np.float32,
)
K = np.matmul(offset, K)
"""

DIST = [0.2, 0.2, 0.2, 0.2, 1]
Kinv = np.linalg.inv(K).astype(np.float32)
K = torch.from_numpy(K).unsqueeze(0).float()
Kinv = torch.from_numpy(Kinv).unsqueeze(0).float()

SN = 80
params = [list(np.linspace(0, DIST[i], SN)) + list(np.linspace(0, -DIST[i], SN)) for i in range(5)]
RTs = []
for i, param in enumerate(params):
  RTs.extend([[p if j == i else 0 for j in range(5)]
    for p in param])
RTs = [get_RT(*p) for p in RTs]

fpaths = glob.glob(f"{DIR}/*.jpg") + glob.glob(f"{DIR}/*.png")
fpaths = [f for f in fpaths if "depth" not in f]
fpaths.sort()
for ind, fpath in enumerate(fpaths):
  if ind % world_size != rank:
    continue
  name = fpath[fpath.rfind("/")+1:-4]
  image = imread_pil(fpath)
  im = transform(image)

  batch = {
    'images' : [im.unsqueeze(0)],
    'cameras' : [{ # This is not real
      'K' : K,#torch.eye(4).unsqueeze(0),
      'Kinv' : Kinv,#torch.eye(4).unsqueeze(0)
    }]
  }

  # Get parameters for the transformation

  pred_imgs = []
  masks = []

  # Generate a new view at the new transformation
  with torch.no_grad():
    # list of (1, 3, 256, 256) [-1, 1]
    BS = 40
    for i in tqdm(range(len(RTs) // BS)):
      img, mask = model_to_test.model.module.forward_angle(
        batch, RTs[i * BS : (i + 1) * BS], get_mask=True)

      masks.extend(mask)
      pred_imgs.extend(img)

    # (1, 1, 256, 256)
    depth = nn.Sigmoid()(model_to_test.model.module.pts_regressor(
      batch['images'][0].cuda()))
    depth = depth.clamp(max=0.04)
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = torch.cat([depth] * 3, 1)
    vutils.save_image(depth, f"{DIR}/{name}_depth.png")

  for i, (img, mask) in enumerate(zip(pred_imgs, masks)):
    vutils.save_image((img + 1) / 2, f"{args.outdir}/{name}_transform{i:03d}.png")
    mask = torch.cat([mask] * 3, 1).float()
    vutils.save_image(mask, f"{args.outdir}/{name}_mask{i:03d}.png")
