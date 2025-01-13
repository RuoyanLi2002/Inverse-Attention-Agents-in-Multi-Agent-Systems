import os
import pickle
import functools
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from Algorithms.BallSDE import marginal_prob_std, diffusion_coeff

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_target_score(num_objs, max_action):
    diffusion_coeff_func = functools.partial(diffusion_coeff, sigma=25)
    tar_path = 'score_wall1_fc.pt'
    with open(tar_path, 'rb') as f:
        score_target = pickle.load(f)
    return TargetScore(score_target.to(device), num_objs, max_action), diffusion_coeff_func

class TargetScore:
    def __init__(self, score, num_objs, max_vel, sigma=25):
        self.score = score
        self.num_objs = num_objs
        self.max_vel = max_vel
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    def get_score(self, x, t0, is_numpy=True, is_norm=True, empty=False):
        if not empty:
            if not torch.is_tensor(x):
                x = torch.tensor(x)
            x = x.to(device)
            bs = x.shape[0]
            t = torch.tensor([t0]*bs).unsqueeze(1).to(device)
            #  model(perturbed_x.float(), random_t, num_objs)
            out_score = self.score(x, t, self.num_objs)
            out_score = out_score.detach()

        else:
            out_score = torch.zeros_like(x).to(device).view(-1, 2)
        return out_score.cpu().numpy() if is_numpy else out_score
    
def visualize(state, gf, count):
  fig, ax = plt.subplots()
  blue_square = patches.Rectangle((-1, -1), 2, 2, color='blue', fill=True)
  red_square = patches.Rectangle((-0.8, -0.8), 1.6, 1.6, color='red', fill=True)

  ax.add_patch(blue_square)
  ax.add_patch(red_square)
  ax.set_aspect('equal', adjustable='box')

  ax.set_xlim(-1.5, 1.5)
  ax.set_ylim(-1.5, 1.5)
  ax.set_xlabel('X-axis')
  ax.set_ylabel('Y-axis')
  ax.set_title('Grid with Centered Squares')

  x = state[0].item()
  y = state[1].item()
  category = state[2].item()

  gf_x = gf[0][0].item()
  gf_y = gf[0][1].item()

  ax.plot(x, y, 'o', color='black')

  color = 'green' if category == 0 else 'red'
  plt.scatter(x, y, color=color)
  plt.quiver(x, y, gf_x, gf_y, scale=5, scale_units='inches')

  plt.grid(True)
  # plt.show()
  
  if not os.path.exists(f"GF_Wall1_FC"):
    os.makedirs(f"GF_Wall1_FC")
  plt.savefig(f'GF_Wall1_FC/gradient_field_wall1_{count}.png')
  
  plt.clf()

num_objs = 1
max_action = 1

target_score, diffusion_coeff_fn = load_target_score(num_objs, max_action)
t0 = 0.01

for i in range(20):
  pos = 2 * torch.rand(2) - 1
  state = torch.cat((pos, torch.tensor([0.0])))
  gf = target_score.get_score(pos, t0=t0, is_norm=True)
  print(gf)
  visualize(state, gf, i)
