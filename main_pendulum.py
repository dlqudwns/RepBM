import numpy as np
import torch
import gym
import os
from ac_pendulum import load_ddpg_agent
from src.config import pendulum_config
from src.train_pipeline_cont import train_pipeline
from src.utils import load_qnet, error_info
from collections import deque

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", help="condor pid", default=0, type=int)
    args = parser.parse_args()
    np.random.seed(args.pid)

    env = gym.make("Pendulum-v0")
    config = pendulum_config
    agent = load_ddpg_agent(config)
    agent.actor.eval()
    agent.critic.eval()
    seedvec = np.random.randint(0, config.MAX_SEED, config.sample_num_traj)

    #factual_types = ['hard', 2.5, 2.0, 1.5, 1.0, 0.5]
    factual_types = ['hard']
    methods = ['Baseline'] + ['mse_pi_{}'.format(ft) for ft in factual_types] +\
              ['repbm_{}'.format(ft) for ft in factual_types]

    num_method = len(methods)
    max_name_length = len(max(methods,key=len))

    mse = [deque() for method in methods]
    ind_mse = [deque() for method in methods]

    results, target = train_pipeline(env, config, agent, factual_types, seedvec)
    for i_method in range(num_method):
        mse_1, mse_2 = error_info(results[i_method], target, methods[i_method].ljust(max_name_length))
        mse[i_method].append(mse_1)
        ind_mse[i_method].append(mse_2)
    print(mse)
    print(ind_mse)
    np.save('results/result_pendulum_{}.txt'.format(args.pid), [mse, ind_mse])
