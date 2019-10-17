import random
from collections import deque

from tqdm import tqdm
import env
import os
import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.memory import ReplayMemory, Transition
from src.models_cont import Critic, Actor
from src import config

# if gpu is to be used
use_cuda = False #torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class DDPGAgent():
    def __init__(self, config):
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.max_action = config.max_action
        self.config = config

        self.actor = Actor(config)
        self.actor_target = Actor(config)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.ac_alpha)

        self.critic = Critic(config)
        self.critic_target = Critic(config)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.ac_alpha)

        self.memory = ReplayMemory(config.buffer_capacity)

    def epsilon_act(self, state, epsilon=None):
        epsilon = epsilon or self.config.ac_epsilon
        sample = random.random()
        if sample > epsilon:
            return self.act(state)
        else:
            return self.uniform_random_act()

    def uniform_random_act(self):
        return np.array([(random.random() * 2 - 1) * self.max_action for _ in range(self.action_dim)])

    def act(self, state):
        """Returns actions for given state as per current policy."""
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        return np.reshape(np.clip(action, -1, 1) * self.max_action, [-1])

    def step(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done, None)

    def learn(self):
        if self.config.ac_batch_size > len(self.memory):
            return

        experiences = self.memory.sample(self.config.ac_batch_size)
        batch = Transition(*zip(*experiences))
        next_states = torch.cat(batch.next_state)
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action) / self.max_action
        rewards = torch.cat(batch.reward).view([actions.shape[0], 1])
        dones = torch.cat(batch.done).view([actions.shape[0], 1])

        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (self.config.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor, self.actor_target)

    def soft_update(self, source, target):
        for target_param, local_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.config.ac_tau * local_param.data
                                    + (1.0 - self.config.ac_tau) * target_param.data)

def epsilon_decay_per_ep(epsilon, config):
    return max(config.ac_epsilon_min, epsilon*config.ac_epsilon_decay)

def preprocess_state(state, state_dim):
    return Tensor(np.reshape(state, [1, state_dim]))

def save_ddpg_agent(ddpg_agent, checkpoint='target_policies',filename='ddpg_agent.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}"
              .format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    state = {'critic_state_dict': ddpg_agent.critic.state_dict(), 'actor_state_dict': ddpg_agent.actor.state_dict()}
    torch.save(state, filepath)

def load_ddpg_agent(config, checkpoint='target_policies',filename='ddpg_agent.pth.tar'):
    agent = DDPGAgent(config)
    filepath = os.path.join(checkpoint, filename)
    if not os.path.exists(filepath):
        raise("No best model in path {}".format(checkpoint))
    checkpoint = torch.load(filepath)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    return agent

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", help="name of the env to train", default='cartpole')
    args = parser.parse_args()
    if args.env_name == 'pendulum':
        env = gym.make("Pendulum-v0")
        config = config.pendulum_config
    elif args.env_name == 'cartpole':
        env = gym.make("ContinuousCartPole-v0")
        config = config.contcartpole_config
    epsilon = config.ac_epsilon
    dp_scores = deque(maxlen=30)
    agent = DDPGAgent(config)

    for i_episode in tqdm(range(config.ac_num_episodes)):
        # Initialize the environment and state
        state = preprocess_state(env.reset(), config.state_dim)
        done = False
        while not done:
            # Select and perform an action
            action = agent.epsilon_act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state, config.state_dim)
            agent.step(state, Tensor([action]), next_state, Tensor([reward]), Tensor([done]))
            # Move to the next state
            state = next_state
        state = preprocess_state(env.reset(), config.state_dim)
        done = False
        reward_sum = 0
        while not done:
            # Select and perform an action
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state, config.state_dim)
            state = next_state
            reward_sum += reward
        dp_scores.append(reward_sum)
        mean_score = np.mean(dp_scores)

        # Set the value we want to achieve
        if mean_score >= config.win_score and i_episode >= 300:
            print('Ran {} episodes. Solved after {} trials ✔'.format(i_episode, i_episode - 30))
            break
        if i_episode % 100 == 0:
            print('[Episode {}] - Averaged sum of rewards over last 100 episodes was {} ticks. Epsilon is {}'
                  .format(i_episode, mean_score, epsilon))
        agent.learn()
        epsilon = epsilon_decay_per_ep(epsilon, config)
    save_ddpg_agent(agent, filename=args.env_name + '_ddpg_agent.pth.tar')

    groundtruth = deque()
    for i_episode in tqdm(range(1000)):
        true_state = preprocess_state(env.reset(), config.state_dim)
        true_done = False
        reward_sum = 0
        while not true_done:
            action = agent.act(true_state)
            true_next_state, true_reward, true_done, _ = env.step(action)
            reward_sum += true_reward
            true_state = preprocess_state(true_next_state, config.state_dim)
        groundtruth.append(reward_sum)
    print('True averaged reward is {} with std dev {:.3e}'.format(np.mean(groundtruth),
                                                                np.std(groundtruth) / len(groundtruth)))
