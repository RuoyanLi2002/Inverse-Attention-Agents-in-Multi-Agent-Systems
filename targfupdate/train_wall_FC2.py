import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm, trange
import functools
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Algorithms.BallSDE import marginal_prob_std, diffusion_coeff, loss_mlp, ode_sampler
from Networks.BallSDENet import ScoreMLP

wall_min = -1
wall_max = 1

epsilon = 10**(-5)

num_samples = 10000
radius = 0.1
num_objs = 1

batch_size = 64
workers = 8

lr = 2e-4
beta1 = 0.5
sigma = 25.0
num_epochs = 15000
visualize_freq = 10

num_classes = 1

# Set seed for reproducibility
SEED = 1
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def duplicate_location(x, y, ls_data):
  for data in ls_data:
    if abs(data[0]-x) < epsilon and abs(data[1]-y) < epsilon:
      return True

  return False

def within_region(x, y):
  return -0.8 < x < 0.8 and -0.8 < y < 0.8

def sample_agent(ls_data):
  while True:
    x = np.round(np.random.uniform(wall_min, wall_max, 1), 2)[0]
    y = np.round(np.random.uniform(wall_min, wall_max, 1), 2)[0]

    if not duplicate_location(x, y, ls_data) and within_region(x, y):
      # print(f"{x}, {y}")
      return [x, y]

def load_train_data(filepath):
  with open(filepath, 'rb') as f:
      return pickle.load(f)

def save_train_data(filepath, train_data):
  with open(filepath, "wb") as f:
    pickle.dump(train_data, f)

def load_or_generate_data(filepath):
  if os.path.exists(filepath):
    print("Train data already exists. Loading train data")
    return load_train_data(filepath)
  else:
    print("Generating train data")
    ls_data = []

    with tqdm(total=num_samples) as pbar:
      for i in range(num_samples):
        state = sample_agent(ls_data)
        ls_data.append(state)

        pbar.update(1)

    ls_data = np.array(ls_data)
    dataset = torch.from_numpy(ls_data)

    save_train_data(filepath, dataset)

    return dataset

def visualize_train_data(dataset):
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

  for i in range(len(dataset)):
    x = dataset[i][0]
    y = dataset[i][1]

    ax.plot(x, y, 'o', color='black')

  plt.grid(True)
  plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_or_generate_data("train_wall2_fc.pkl")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
score = ScoreMLP(marginal_prob_std_fn, n_box = None, mode = None , device = device)
score.to(device)

optimizer = optim.Adam(score.parameters(), lr=lr, betas=(beta1, 0.999))


print("Starting Training Loop...")
for epoch in trange(num_epochs):
  for i, real_data in enumerate(dataloader):
    real_data = real_data.to(device)
    # loss = loss_fn_state(score, real_data, marginal_prob_std_fn, num_objs=num_objs)
    loss = loss_mlp(score, None, real_data, marginal_prob_std_fn, num_objs = num_objs, eps=1e-5)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
      if (epoch + 1) % visualize_freq == 0:
        with open('score_wall2_fc.pt', 'wb') as f:
          pickle.dump(score, f)


    optimizer.zero_grad()

