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
from models.base_model import BaseModel
from data.simple import SimpleDataset
from utils.misc import imread_pil, imwrite, torch2numpy
from utils.visualize import heatmap_numpy, VideoWriter
from options.options import get_model

#ffmpeg -i identity_intrinsic.mp4 -vf "[in] pad=2*iw:ih [left]; movie=wrong_intrinsic.mp4 [right]; [left][right] overlay=main_w/2:0 [out]" -b:v 16000k Output.mp4

parser = argparse.ArgumentParser()
parser.add_argument('--model-path',
  default='./modelcheckpoints/realestate/synsin.pth')
parser.add_argument('--indir',
  default='stylegan2_bedroom_synthesis')
args = parser.parse_args()
torch.backends.cudnn.enabled = True

# synsin is z buffer model
MODEL_PATH = args.model_path
BATCH_SIZE = 4
device = 'cuda'
DIR = args.indir #'/home/b146466/myvideo/frames/1_translation_down/'

# the option is stored in the pth file
opts = torch.load(MODEL_PATH)['opts']
opts.render_ids = [1]

model = get_model(opts)

if 'sync' in opts.norm_G:
  model = convert_model(model)
model = nn.DataParallel(model).cuda()

model_to_test = BaseModel(model, opts)
model_to_test.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
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

DIST = 0.1
Kinv = np.linalg.inv(K).astype(np.float32)
K = torch.from_numpy(K).unsqueeze(0).float()
Kinv = torch.from_numpy(Kinv).unsqueeze(0).float()

fpaths = glob.glob(f"{DIR}/*.jpg") + glob.glob(f"{DIR}/*.png")
fpaths = [f for f in fpaths if "depth" not in f]
fpaths.sort()
for fpath in fpaths:
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
  SN = 48
  params = [circle(DIST, SN) for _ in range(5)]
  RTs = []
  for i, param in enumerate(params):
    RTs.extend([[p if j == i else 0 for j in range(5)]
      for p in param])
  RTs = [get_RT(*p) for p in RTs]

  # video_writer = VideoWriter(f"{DIR}/{name}_transformed.mp4", 256, 256)
  pred_imgs = []
  # Generate a new view at the new transformation
  with torch.no_grad():
    # list of (1, 3, 256, 256) [-1, 1]
    BS = SN

    for i in tqdm(range(len(RTs) // 4)):
      res = model_to_test.model.module.forward_angle(
        batch, RTs[i * BS : (i + 1) * BS])
      pred_imgs.extend(res)
      #for r in res:
      #  img = torch2numpy((r.clamp(-1, 1) + 1) / 2 * 255).astype("int8")
      #  img = img[0].transpose(1, 2, 0)
      #  print(img.shape)
      #  video_writer.write(img)

    # (1, 1, 256, 256)
    depth = nn.Sigmoid()(model_to_test.model.module.pts_regressor(
      batch['images'][0].cuda()))
    depth = depth.clamp(max=0.04)
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = torch.cat([depth] * 3, 1)
    #heatmap = heatmap_numpy(torch2numpy(depth)[0])
    #heatmap = torch.from_numpy(heatmap).permute(0, 3, 1, 2)
    #print(heatmap.shape, heatmap.min(), heatmap.max())
    vutils.save_image(depth, f"{DIR}/{name}_depth.png")

  #vutils.save_image((im.unsqueeze(0) + 1) / 2, "original.png")
  os.system("rm temp/*.png")
  for i, img in enumerate(pred_imgs):
    vutils.save_image((img + 1) / 2, "temp/transform%03d.png" % i)
  os.system(f"ffmpeg -y -f image2 -i temp/transform%03d.png -pix_fmt yuv420p -b:v 16000k {DIR}/{name}_transformed.mp4")