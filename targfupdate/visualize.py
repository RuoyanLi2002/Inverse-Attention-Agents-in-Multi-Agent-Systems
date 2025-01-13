import numpy as np
from matplotlib import pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import functools
from tqdm import trange
import pickle
from Algorithms.BallSDE import marginal_prob_std, diffusion_coeff
import io
SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
        
def load_target_score(num_objs, max_action):
    diffusion_coeff_func = functools.partial(diffusion_coeff, sigma=25)
    tar_path = 'score_wolf2.pt'
    with open(tar_path, 'rb') as f:
        score_target = CPU_Unpickler(f).load()
    return TargetScore(score_target.to(device), num_objs, max_action), diffusion_coeff_func

class TargetScore:
    def __init__(self, score, num_objs, max_vel, sigma=25):
        self.score = score
        self.num_objs = num_objs
        self.max_vel = max_vel
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    def get_score(self, state_inp, t0, is_numpy=True, is_norm=True, empty=False):
        if not empty:
            if not torch.is_tensor(state_inp):
                state_inp = torch.tensor(state_inp)
            positions = state_inp.view(-1, self.num_objs, 3).to(device)[:, :, :2]
            categories = state_inp.view(-1, self.num_objs, 3).to(device)[:, :, -1:]
            bs = positions.shape[0]
            positions = positions.view(-1, 2).float()
            categories = categories.view(-1).long()
            samples_batch = torch.tensor([i for i in range(bs) for _ in range(self.num_objs)], dtype=torch.int64).to(device)
            edge_index = knn_graph(positions, k=self.num_objs-1, batch=samples_batch)
            t = torch.tensor([t0]*bs).unsqueeze(1).to(device)
            inp_data = Data(x=positions, edge_index=edge_index, c=categories)

            out_score = self.score(inp_data, t, self.num_objs)
            out_score = out_score.detach()
            if is_norm:
                out_score = out_score * torch.min(
                    torch.tensor([1, self.max_vel / (torch.max(torch.abs(out_score)) + 1e-7)]).to(device))
            else:
                out_score = out_score
        else:
            out_score = torch.zeros_like(state_inp).to(device).view(-1, 2)
        return out_score.cpu().numpy() if is_numpy else out_score
    

num_objs = 2
max_action = 1

target_score, diffusion_coeff_fn = load_target_score(num_objs, max_action)

state = torch.tensor([0.0, 0.0, 1.0,
                      1.0, 1.0, 1.0])
t0 = 0.01

gf = target_score.get_score(state, t0=t0, is_norm=True)
print(gf)

import pickle
with open('gf.pkl', 'wb') as file: 
    pickle.dump(gf, file) 

state = state.reshape(num_objs, 3)

for i in range(num_objs):
    x = state[i][0].item()
    y = state[i][1].item()
    category = state[i][2].item()

    gf_x = gf[i][0].item() if category==1 else -gf[i][0].item()
    gf_y = gf[i][1].item() if category==1 else -gf[i][1].item()

    gf_x = gf_x * 3
    gf_y = gf_y * 3
    print(f"x: {x}; y: {y}; gf_x: {gf_x}; gf_y: {gf_y}")
    color = 'green' if category == 0 else 'red'
    plt.scatter(x, y, color=color)
    plt.quiver(x, y, gf_x, gf_y, scale=5, scale_units='inches')

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.savefig('gradient_field.png')
plt.show()

