import os
import sys
import time
import pickle
import cv2
import gym
import ebor
from ipdb import set_trace
import numpy as np
from tqdm import tqdm, trange
import argparse
import functools
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.utils import make_grid

from torch_geometric.nn import knn_graph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from ball_utils import exists_or_mkdir, save_video
from Algorithms.BallSDE import marginal_prob_std, diffusion_coeff, loss_fn_state, ode_sampler, loss_mlp
from Networks.BallSDENet import ScoreModelGNN, ScoreMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def snapshot(env, file_name):
    img = env.render(img_size=256)
    cv2.imwrite(file_name, img)

def collect_data(env, env_name, num_samples, num_boxes, is_random=False, suffix=''):
    exists_or_mkdir('./ExpertDatasets/')
    debug_path = f'./ExpertDatasets/{env_name}_{suffix}/'
    exists_or_mkdir(debug_path)
    samples = []
    debug_freq = 100
    num_boxes
    
    with tqdm(total=num_samples) as pbar:
        while len(samples) < num_samples:
            cur_state = env.reset(is_random=is_random)
            samples.append(cur_state)
            if len(samples) % debug_freq == 0:
                snapshot(env, f'{debug_path}debug_{len(samples)//debug_freq}.png')
            pbar.update(1)
    samples = env.flatten_states(samples)
    samples = np.stack(samples, axis=0)
    return samples


class MyDataset(Dataset):
    def __init__(self, data, num_objs):
        self.gnndata = data[:,:num_objs*3]
        self.mlpdata = data[:,num_objs*3:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.gnndata[index]

def visualize_states(eval_states, env, logger, nrow, epoch, suffix):
    # states -> images
    # if scores, then try to visualize gradient
    imgs = []
    for obj_state in eval_states:
        obj_state = obj_state.detach().cpu().numpy()
        obj_state = env.unflatten_states([obj_state])[0]
        env.set_state(obj_state)
        img = env.render(img_size=render_size)
        imgs.append(img)
    batch_imgs = np.stack(imgs, axis=0)
    ts_imgs = torch.tensor(batch_imgs).permute(0, 3, 1, 2)
    grid = make_grid(ts_imgs.float(), padding=2, nrow=nrow, normalize=True)
    logger.add_image(f'Images/dynamic_{suffix}', grid, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # file
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--sigma', type=float, default=25.)
    # env
    parser.add_argument('--pattern', type=str, default='CircleCluster')
    parser.add_argument('--num_per_class', type=int, default=7)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    # train
    parser.add_argument('--n_epoches', type=int, default=10000)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--t0', type=float, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--workers', type=int, default=8)
    # eval
    parser.add_argument('--visualize_freq', type=int, default=10)
    parser.add_argument('--video_freq', type=int, default=100)
    parser.add_argument('--test_num', type=int, default=16)
    parser.add_argument('--test_col', type=int, default=4)
    parser.add_argument('--render_size', type=int, default=256)

    # load args
    args = parser.parse_args()
    num_per_class = args.num_per_class
    num_classes = args.num_classes
    num_objs = num_per_class*num_classes
    n_samples = args.n_samples
    batch_size = args.batch_size
    workers = args.workers
    lr = args.lr
    beta1 = args.beta1
    render_size = args.render_size
    # dataset_path = f'./ExpertDatasets/{args.data_name}.pth'
    dataset_path = f'./ExpertDatasets/distill_alltogether_6.pkl'
    # dataset_path = f'./ExpertDatasets/alltogether_6.pkl'
    # dataset_path = f'./ExpertDatasets/cluster3.pkl'
    # dataset_path = f'./ExpertDatasets/randomblue.pkl'
    # dataset_path = f'./ExpertDatasets/noshuffle_cluster3.pkl'
    # dataset_path = f'./ExpertDatasets/lineup.pkl'
    # dataset_path = f'./ExpertDatasets/lineup_fixed_onedirection.pkl'
    # create log path
    exists_or_mkdir('./logs')
    ckpt_path = f'./logs/{args.log_dir}/'
    exists_or_mkdir(ckpt_path)
    eval_path = f'./logs/{args.log_dir}/test_batch/'
    exists_or_mkdir(eval_path)
    tb_path = f'./logs/{args.log_dir}/tb'
    exists_or_mkdir(tb_path)
    writer = SummaryWriter(tb_path)

    ''' init my env '''
    # env_name = '{}-{}Ball{}Class-v0'.format(args.pattern, num_objs, num_classes)
    # env = gym.make(env_name)
    # env.seed(args.seed)
    # env.reset()

    ''' Prepare training data '''
    if os.path.exists(dataset_path):
        print('### found existing dataset ###')
        with open(dataset_path, 'rb') as f:
            data_samples = pickle.load(f)
        dataset = torch.tensor(data_samples)
        dataset = dataset[: n_samples]
    else:
        print('### not found existding dataset, start collecting data ###')
        ts = time.time()
        data_samples = collect_data(env, env_name, n_samples, num_per_class, is_random=False, suffix=args.data_name)
        with open(dataset_path, 'wb') as f:
            pickle.dump(data_samples, f)
        dataset = torch.tensor(data_samples)
        print('### data collection done! takes {:.2f} to collect {} samples ###'.format(time.time() - ts, n_samples))
    
    ''' Prepare dataloader '''
    print(f'Dataset Size: {len(dataset)}')
    print(f'Dataset Shape: {dataset.shape}')
    print(dataset[:5,:])
    dataset = dataset.reshape(-1, dataset.shape[-1])

    # prepare graph-based dataset
    k = num_objs - 1 # fully connected graph
    # edge = knn_graph(dataset[0].reshape(num_objs, 2+1)[:, :2], k, loop=False)
    # dataset = list(map(lambda x: Data(x=x[:, :2].float(),  edge_index=edge, c=x[:, -1].long()), dataset.reshape(dataset.shape[0], num_objs, 2+1)))
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    dataset = MyDataset(dataset,num_objs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    ''' Init Model '''
    # init SDE-related params
    sigma = args.sigma  # @param {'type':'number'}
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
    # score = ScoreModelGNN(marginal_prob_std_fn, num_classes=num_classes, device=device)
    score = ScoreMLP(marginal_prob_std_fn, num_classes=num_classes, device=device)
    score.to(device)

    optimizer = optim.Adam(score.parameters(), lr=lr, betas=(beta1, 0.999))

    num_epochs = args.n_epoches
    print("Starting Training Loop...")
    for epoch in trange(num_epochs):
        # For each batch in the dataloader
        # for i in range(1000):
        for i, real_data in enumerate(dataloader):
            # augment data first
            real_data = real_data.to(device)
            # calc score-matching loss
            # loss = loss_fn_state(score, real_data, marginal_prob_std_fn, num_objs=num_objs)
            loss = loss_mlp(score, real_data, marginal_prob_std_fn, num_objs=num_objs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add writer
            writer.add_scalar('train_loss', loss, i + epoch*len(dataloader))
        with torch.no_grad():
            if (epoch + 1) % args.visualize_freq == 0:
                
                with open(ckpt_path + f'score.pt', 'wb') as f:
                    pickle.dump(score, f)
        optimizer.zero_grad()

    env.close()

