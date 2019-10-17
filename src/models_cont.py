import torch
import torch.optim as optim
import torch.nn as nn
from collections import namedtuple
from src.memory import SampleSet
from src.utils import *

use_cuda = False #torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        state_dim, action_dim, hidden_dims = config.state_dim, config.action_dim, config.ac_hidden_dims
        mlp_layers = []
        prev_hidden_size = state_dim + action_dim
        for next_hidden_size in hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        mlp_layers.append(
            nn.Linear(prev_hidden_size, 1)
        )
        self.model = nn.Sequential(*mlp_layers)

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=1))

class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        state_dim, action_dim, hidden_dims = config.state_dim, config.action_dim, config.ac_hidden_dims
        mlp_layers = []
        prev_hidden_size = state_dim
        for next_hidden_size in hidden_dims + [action_dim]:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        self.model = nn.Sequential(*mlp_layers)

    def forward(self, state):
        return self.model(state)

FactualTransition = namedtuple(
    'FactualTransition', ['state', 'action', 'state_diff', 'reward', 'done', 'time', 'factual', 'last_factual'])

class SmallSampleSet(object):

    def __init__(self, config, factual_types):
        self.max_action = config.max_action
        self.max_len = config.max_length
        self.num_episode = 0
        self.u = np.zeros(self.max_len) # u_{0:t}
        self.memory = [[] for h in range(self.max_len)]
        self.terminal = []
        self.total_factuals = Tensor(np.zeros([self.max_len, len(factual_types)]))
        self.factual_types = factual_types

    def push(self, *args):
        t = FactualTransition(*args)
        self.memory[t.time].append(t)
        self.total_factuals[t.time] += t.factual[0]
        if t.done and t.time < self.max_len-1:
            self.terminal.append(t)

    def standardize_state(self, state):
        return (state - self.state_mean) / self.state_std
    def unstandardize_state_diff(self, stan_state_diff):
        return stan_state_diff * self.state_diff_std + self.state_diff_mean
    def unstandardize_reward(self, stan_reward):
        return stan_reward * self.reward_std + self.reward_mean

    def update_statistics(self, statistics=None):
        self.num_episode = len(self.memory[0])
        self.u = self.total_factuals/self.num_episode
        self.allsamples = [item for sublist in self.memory for item in sublist]

        if statistics is None:
            states = torch.stack([t.state for t in self.allsamples])
            state_diffs = torch.stack([t.state_diff for t in self.allsamples])
            rewards = torch.stack([t.reward for t in self.allsamples])
            self.state_mean, self.state_std = torch.mean(states, axis=0), torch.std(states, axis=0) + 1e-3
            self.state_diff_mean, self.state_diff_std = torch.mean(state_diffs, axis=0), torch.std(state_diffs, axis=0) + 1e-3
            self.reward_mean, self.reward_std = torch.mean(rewards, axis=0), torch.std(rewards, axis=0) + 1e-3
        else:
            self.state_mean, self.state_std, self.state_diff_mean, self.state_diff_std, \
            self.reward_mean, self.reward_std = statistics

        for t, sublist in enumerate(self.memory):
            for i, item in enumerate(sublist):
                new_state = (item.state - self.state_mean) / self.state_std
                new_state_diff = (item.state_diff - self.state_diff_mean) / self.state_diff_std
                new_reward = (item.reward - self.reward_mean) / self.reward_std
                new_action = item.action / self.max_action
                self.memory[t][i] = FactualTransition(
                    new_state, new_action, new_state_diff, new_reward,
                    item.done, item.time, item.factual, item.last_factual)

        return (self.state_mean, self.state_std, self.state_diff_mean, self.state_diff_std,
                self.reward_mean, self.reward_std)

    def sample(self, batch_size):
        while True:
            time = random.randint(0,self.max_len-1)
            if len(self.memory[time]) >= batch_size:
                return random.sample(self.memory[time], batch_size)

    def sample_terminal(self, batch_size):
        if len(self.terminal) >= batch_size:
            return random.sample(self.terminal, batch_size)
        else:
            return self.terminal

    def __len__(self):
        return len(self.memory[0])

class MDPnet(nn.Module):
    def __init__(self, config, loss_str, factual_index):
        super(MDPnet, self).__init__()
        self.config = config
        self.loss_str = loss_str
        self.factual_index = factual_index
        # representation
        mlp_layers = []
        prev_hidden_size = config.state_dim
        for next_hidden_size in config.rep_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        self.representation = nn.Sequential(*mlp_layers)
        self.rep_dim = prev_hidden_size

        # Transition
        mlp_layers = []
        prev_hidden_size = self.rep_dim + config.action_dim
        for next_hidden_size in config.transition_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        mlp_layers.append(
            nn.Linear(prev_hidden_size, config.state_dim)
        )
        self.transition = nn.Sequential(*mlp_layers)

        #Reward
        mlp_layers = []
        prev_hidden_size = self.rep_dim + config.action_dim
        for next_hidden_size in config.reward_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        mlp_layers.append(
            nn.Linear(prev_hidden_size, 1)
        )
        self.reward = nn.Sequential(*mlp_layers)

    def forward(self, state, action):
        rep = self.representation(state)
        next_state_diff = self.transition(torch.cat([rep, action], dim=1)).view(-1, self.config.state_dim)
        reward = self.reward(torch.cat([rep, action], dim=1)).view(-1, 1)
        return next_state_diff, reward, rep

    def fit(self, memory, dev_memory):
        print('Learn {}, {} mdp model'.format(self.loss_str, self.factual_index))  # mse_pi
        self.standardize_state = memory.standardize_state
        self.unstandardize_state_diff = memory.unstandardize_state_diff
        self.unstandardize_reward = memory.unstandardize_reward

        lr = self.config.lr
        best_train_loss = 100
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.config.weight_decay)
        for i_episode in range(self.config.train_num_episodes):
            train_losses = []
            dev_losses = []
            for i_batch in range(self.config.train_num_batches):
                train_losses.append(self.step(memory, optimizer))
                dev_losses.append(self.step(dev_memory))
            train_loss, dev_loss = np.mean(train_losses), np.mean(dev_losses)
            if (i_episode + 1) % self.config.print_per_epi == 0:
                print('Episode {:0>3d}: train loss {:.3e}, dev loss {:.3e}, learning_rate {:.2e}'.format(i_episode + 1, train_loss, dev_loss, lr))
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                else:
                    lr *= self.config.lr_decay
                    learning_rate_update(optimizer, lr)

    def predict(self, state, action):
        # gets unstandardized input and gives unstandardized output
        stan_state, stan_action = self.standardize_state(state), action / self.config.max_action
        state_diff, reward, _ = self.forward(stan_state, stan_action)
        return self.unstandardize_state_diff(state_diff), self.unstandardize_reward(reward)

    def step(self, memory, optimizer=None):
        if optimizer is not None:
            self.train()
            batch_size = self.config.train_batch_size
        else:
            self.eval()
            batch_size = self.config.test_batch_size

        if batch_size > len(memory):
            return

        transitions = memory.sample(self.config.train_batch_size)
        time = transitions[0].time
        ratio = 1/memory.u[time, self.factual_index] if memory.u[time, self.factual_index] != 0 else 0

        batch = FactualTransition(*zip(*transitions))
        states, states_diff, actions, reward, factual, last_factual = \
            [torch.cat(each) for each in
             [batch.state, batch.state_diff, batch.action, batch.reward, batch.factual, batch.last_factual]]
        factual, last_factual = factual[:, self.factual_index], last_factual[:, self.factual_index]

        mse_pi_weights = factual * ratio
        repbm_weights = mse_pi_weights + 1
        predict_state_diff, predict_reward_value, rep = self.forward(states.type(Tensor), actions.type(Tensor))

        if torch.sum(factual) > 0 and torch.sum(last_factual) > 0:
            factual_mean_rep = (rep * factual[:, None]).sum(0) / factual.sum()
            control_mean_rep = (rep * last_factual[:, None]).sum(0) / factual.sum()
            loss_rep = ((factual_mean_rep - control_mean_rep)**2).sum(0)
        else:
            loss_rep = 0

        # Compute loss
        if self.loss_str == 'mse_mu': # MSE_mu
            loss = torch.nn.MSELoss()(predict_state_diff, states_diff) \
                   + torch.nn.MSELoss()(predict_reward_value, reward)
        elif self.loss_str == 'repbm': # MSE_pi + MSE_mu (objective in RepBM paper)
            loss = weighted_mse_loss(predict_state_diff, states_diff, repbm_weights) \
                   + weighted_mse_loss(predict_reward_value, reward, repbm_weights) \
                   + self.config.alpha_rep * loss_rep #+ config.alpha_bce*torch.nn.BCELoss(weights)(predict_soft_done,done)
        elif self.loss_str == 'mse_pi': # MSE_pi: only MSE_pi, as an ablation study
            loss = weighted_mse_loss(predict_state_diff, states_diff, mse_pi_weights) \
                   + weighted_mse_loss(predict_reward_value, reward, mse_pi_weights)

        if optimizer is not None:
            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            optimizer.step()
        return loss.item()


class TerminalClassifier(nn.Module):
    def __init__(self, config):
        super(TerminalClassifier, self).__init__()
        self.config = config

        mlp_layers = []
        prev_hidden_size = self.config.state_dim #self.config.rep_hidden_dims[-1]
        for next_hidden_size in config.terminal_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        mlp_layers.extend([
            nn.Linear(prev_hidden_size, 1),
        ])
        self.terminal = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.terminal(x)

    def predict(self, state):
        return self.terminal(self.standardize_state(state))

    def fit(self, memory, dev_memory):
        print('Learn terminal classifier')
        self.standardize_state = memory.standardize_state
        lr = 0.01
        best_train_acc = 0
        optimizer = optim.Adam([param for param in self.parameters() if param.requires_grad], lr=lr)
        for i_episode in range(self.config.tc_num_episode):
            train_losses = []; train_acces = []
            dev_losses = []; dev_acces = []
            for i_batch in range(self.config.tc_num_batches):
                train_loss_batch, train_acc_batch = self.step(memory, optimizer)
                dev_loss_batch, dev_acc_batch = self.step(dev_memory)
                train_losses.append(train_loss_batch); train_acces.append(train_acc_batch)
                dev_losses.append(dev_loss_batch); dev_acces.append(dev_acc_batch)
            train_loss, dev_loss = np.mean(train_losses), np.mean(dev_losses)
            train_acc, dev_acc = np.mean(train_acces), np.mean(dev_acces)
            if (i_episode+1) % self.config.print_per_epi == 0:
                print('Episode {:0>3d}: train loss {:.3e} acc {:.3e}, dev loss {:.3e} acc {:.3e}'.
                      format(i_episode + 1, train_loss, train_acc, dev_loss, dev_acc))
            if train_acc > best_train_acc:
                best_train_acc = train_acc
            else:
                lr *= 0.9
                learning_rate_update(optimizer, lr)

    def step(self, memory, optimizer=None):
        if optimizer is not None:
            self.train()
            batch_size = self.config.tc_batch_size
        else:
            self.eval()
            batch_size = self.config.tc_test_batch_size
        transitions_origin = memory.sample(batch_size)
        transitions_terminal = memory.sample_terminal(batch_size)
        transitions = transitions_origin + transitions_terminal

        states = torch.cat([t.state for t in transitions])
        done = torch.cat([FloatTensor([[t.done and t.time < self.config.max_length - 1]]) for t in transitions])
        predict_soft_done = self.forward(states.type(Tensor))

        loss = torch.nn.BCEWithLogitsLoss()(predict_soft_done, done)
        acc = ((predict_soft_done.detach()[:,0]>0).float()==done[:,0]).sum().item()/(done.size()[0])
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.item(), acc
