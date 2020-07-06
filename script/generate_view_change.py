import matplotlib.pyplot as plt
import quaternion
import numpy as np

import os
os.environ['DEBUG'] = '0'

import torch
import torch.nn as nn
import torchvision.transforms as transforms


from models.networks.sync_batchnorm import convert_model
from models.base_model import BaseModel
from data.simple import SimpleDataset
from utils.misc import imread_pil
from options.options import get_model

torch.backends.cudnn.enabled = True

# synsin is z buffer model
MODEL_PATH = './modelcheckpoints/realestate/synsin.pth'
BATCH_SIZE = 1
device = 'cuda'
fpath = '../../myvideo/frames/1_translation_down/0001.png'

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
    #transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def get_RT(theta, phi, tx, ty, tz):
  RT = torch.eye(4).unsqueeze(0)
  # Set up rotation
  RT[0,0:3,0:3] = torch.Tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_vector([phi, theta, 0])))
  # Set up translation
  RT[0,0:3,3] = torch.Tensor([tx, ty, tz])
  return RT

# Parameters for the transformation
theta = -0.15
phi = -0.1
tx = 0
ty = 0
tz = 0

RTs = [get_RT(theta, phi, tx, ty, tz)
  for tz in np.linspace(-0.2, 0.2, 48)]

image = imread_pil(fpath)
im = transform(im)

batch = {
    'images' : [im.unsqueeze(0)],
    'cameras' : [{ # This is not real
        'K' : torch.eye(4).unsqueeze(0),
        'Kinv' : torch.eye(4).unsqueeze(0)
    }]
}

# Generate a new view at the new transformation
with torch.no_grad():
    pred_imgs = model_to_test.model.module.forward_angle(batch, RTs)
    depth = nn.Sigmoid()(model_to_test.model.module.pts_regressor(batch['images'][0].cuda()))