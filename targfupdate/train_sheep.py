import os
import random
import pickle
import numpy as np
from tqdm import tqdm, trange
import functools
import torch
import torch.optim as optim
from torch_geometric.nn import knn_graph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from Algorithms.BallSDE import marginal_prob_std, diffusion_coeff, loss_fn_state, ode_sampler
from Networks.BallSDENet import ScoreModelGNN


min_x = -1
max_x = 1
min_y = -1
max_y = 1
epsilon = 10**(-5)

num_samples = 10000
min_radius = 0.5
max_radius = 2
num_wolf = 3
num_objs = num_wolf + 1

batch_size = 8
workers = 8

lr = 2e-4
beta1 = 0.5
sigma = 25.0
num_epochs = 15000
visualize_freq = 10

num_classes = 2

# Set seed for reproducibility
SEED = 1
random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def within_grid(x, y):
  return x >= min_x and x <= max_x and y >= min_y and y <= max_y

def duplicate_location(x, y, ls_data):
  for data in ls_data:
    if abs(data[0]-x) < epsilon and abs(data[1]-y) < epsilon:
      return True

  return False

def sample_sheep(ls_data):
  while True:
    x = np.round(np.random.uniform(min_x, max_x, 1), 2)[0]
    y = np.round(np.random.uniform(min_y, max_y, 1), 2)[0]

    if not duplicate_location(x, y, ls_data):
      return [x, y, 0.0]


def sample_wolf(min_radius, max_radius, num_wolf, sheep_location):
  ls_wolf_states = []

  for i in range(num_wolf):
    while True:
      x = random.choice([-1, 1]) * np.random.uniform(min_radius, max_radius, 1)[0] + sheep_location[0]
      y = random.choice([-1, 1]) * np.random.uniform(min_radius, max_radius, 1)[0] + sheep_location[1]
      if within_grid(x, y):
        ls_wolf_states.append(x)
        ls_wolf_states.append(y)
        ls_wolf_states.append(np.array(1.0))
        break

  return ls_wolf_states

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
        sheep_states = sample_sheep(ls_data)
        wolf_states = sample_wolf(min_radius, max_radius, num_wolf, sheep_states)
        ls_data.append(sheep_states + wolf_states)

        pbar.update(1)

    ls_data = np.array(ls_data)
    dataset = torch.from_numpy(ls_data)

    save_train_data(filepath, dataset)

    return dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_or_generate_data("train_sheep.pkl")
k = num_objs - 1

edge = knn_graph(dataset[0].reshape(num_objs, 2+1)[:, :2], k, loop=False)
dataset = list(map(lambda x: Data(x=x[:, :2].float(),  edge_index=edge, c=x[:, -1].long()), dataset.reshape(dataset.shape[0], num_objs, 2+1)))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)


marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
score = ScoreModelGNN(marginal_prob_std_fn, num_classes=num_classes, device=device)
score.to(device)

optimizer = optim.Adam(score.parameters(), lr=lr, betas=(beta1, 0.999))


print("Starting Training Loop...")
for epoch in trange(num_epochs):
  for i, real_data in enumerate(dataloader):
    real_data = real_data.to(device)
    loss = loss_fn_state(score, real_data, marginal_prob_std_fn, num_objs=num_objs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
      if (epoch + 1) % visualize_freq == 0:
        with open('score_sheep.pt', 'wb') as f:
          pickle.dump(score, f)


    optimizer.zero_grad()