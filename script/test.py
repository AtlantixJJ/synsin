import sys, os, argparse, glob
sys.path.insert(0, ".")
import numpy as np
from tqdm import tqdm
os.environ["DEBUG"] = ""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils

import syn

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

DIR = args.indir
synsin = syn.SynsinTransformer(args.model_path)

transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop((256, 256)),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

fpaths = glob.glob(f"{DIR}/*.jpg") + glob.glob(f"{DIR}/*.png")
fpaths = [f for f in fpaths if "depth" not in f]
fpaths.sort()
for ind, fpath in enumerate(fpaths):
  if ind % world_size != rank:
    continue
  name = fpath[fpath.rfind("/")+1:-4]
  image = imread_pil(fpath)
  im = transform(image)

  # Get parameters for the transformation
  pred_imgs, masks = synsin(im)
  
  for i, (img, mask) in enumerate(zip(pred_imgs, masks)):
    vutils.save_image((img + 1) / 2, f"{args.outdir}/{name}_transform{i:03d}.png")
    mask = torch.cat([mask] * 3, 1).float()
    vutils.save_image(mask, f"{args.outdir}/{name}_mask{i:03d}.png")
