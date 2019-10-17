
from tqdm import tqdm
from timeit import default_timer as timer
from src.models_cont import MDPnet, TerminalClassifier, SmallSampleSet
from src.utils import *
from collections import deque
import torch.optim as optim

# if gpu is to be used
use_cuda = False #torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class benchmark(object):

    def __init__(self, msg, fmt="%0.3g"):
        self.msg = msg
        self.fmt = fmt

    def __enter__(self):
        print('entering: ' + self.msg)
        self.start = timer()
        return self

    def __exit__(self, *args):
        t = timer() - self.start
        print(("%s : " + self.fmt + " seconds") % (self.msg, t))
        self.time = t

def rollout_batch(init_states, mdpnet, terminal_classifier, num_rollout, egbehavior,
                  epsilon, action_size, maxlength, config):
    num_init_states = init_states.size()[0]
    batch_size = num_init_states * num_rollout
    init_states = init_states.repeat(num_rollout,1)

    states = init_states
    done = torch.zeros([batch_size, 1])

    t_reward = torch.zeros([batch_size, 1])
    for _ in range(maxlength):
        actions = Tensor(egbehavior.act_batch(states, epsilon))

        states_re = states * Tensor(np.float32(config.rescale))

        states_diff, reward, _ = mdpnet.predict(states_re.type(Tensor), actions.type(Tensor))
        next_states = states_diff.detach() / Tensor(np.float32(config.rescale)) + states
        t_reward = t_reward + (1 - done.type(Tensor)).float() * reward.detach()

        states = next_states
        done = done.type(torch.BoolTensor) | (terminal_classifier.predict(states.type(Tensor)).detach()[:, 0] > 0)[:, None]


        if sum(done.type(Tensor)) == batch_size:
            break

    value = np.reshape(t_reward.numpy(), [num_rollout, num_init_states])
    return np.mean(value,0)

class EpsilonGreedyBehavior(object):
    def __init__(self, target_agent):
        self.target_agent = target_agent
    def act(self, state, epsilon):
        if random.random() > epsilon:
            return self.target_agent.act(state)
        else:
            return self.target_agent.uniform_random_act()

    def act_batch(self, states, epsilon):
        batch_size = states.size()[0]
        sample = np.random.random([batch_size,1])
        greedy_a = self.target_agent.act(states)
        random_a = np.random.uniform(
            -1, 1, size=(batch_size, self.target_agent.action_dim)) * self.target_agent.max_action
        return (sample < epsilon) * random_a + (sample >= epsilon) * greedy_a

    def get_action_prob(self, state, action, epsilon):
        target_action = self.target_agent.act(state)
        if action == target_action:
            return (1 - epsilon) #np.inf
        else:
            return epsilon / 2 / self.target_agent.max_action


def train_pipeline(env, config, eval_agent, factual_types, seedvec = None):

    with benchmark('time_sampling') as time_sampling:
        memory = SmallSampleSet(config, factual_types) # same the tuples for model training
        dev_memory = SmallSampleSet(config, factual_types)
        egbehavior = EpsilonGreedyBehavior(eval_agent)

        # fix initial state
        if seedvec is None:
            seedvec = np.random.randint(0, config.MAX_SEED, config.sample_num_traj)

        scores = deque()
        for i_episode in tqdm(range(config.sample_num_traj)):
            # Initialize the environment and state
            env.seed(seedvec[i_episode].item())
            state = Tensor(np.reshape(env.reset(), [1, config.state_dim]) * np.float32(config.rescale))
            reward_sum = 0
            factual = np.ones(len(factual_types))
            for n_steps in range(config.max_length):
                # Select and perform an action
                target_action = eval_agent.act(state)
                behavior_action = egbehavior.act(state, config.behavior_epsilon)

                p_pie = np.ones_like(factual)
                for i, ft in enumerate(factual_types):
                    if ft == 'hard':
                        p_pie[i] = 1.0 if target_action == behavior_action else 0.0
                    else:
                        squared_difference = ((target_action - behavior_action) / config.max_action) ** 2
                        p_pie[i] = torch.exp(-torch.sum(Tensor(squared_difference)) / 2 / (ft ** 2))

                last_factual = factual * (1.0 - p_pie)  # 1{a_{0:t-1}==\pie, a_t != \pie}
                factual = factual * p_pie  # 1{a_{0:t}==\pie}

                next_state, reward, done, _ = env.step(behavior_action)
                next_state = Tensor(np.reshape(next_state, [1, config.state_dim]) * np.float32(config.rescale))
                state_diff = next_state - state

                if i_episode < config.train_num_traj:
                    memory.push(state, Tensor(behavior_action), state_diff, Tensor([reward]), done, n_steps, Tensor([factual]), Tensor([last_factual]))
                else:
                    dev_memory.push(state, Tensor(behavior_action), state_diff, Tensor([reward]), done, n_steps, Tensor([factual]), Tensor([last_factual]))

                state = next_state
                reward_sum += reward
                if done:
                    break
            scores.append(reward_sum)

        statistics = memory.update_statistics()
        dev_memory.update_statistics(statistics)
        print('Sampling {} trajectories, the mean survival time is {}'.format(config.sample_num_traj, np.mean(scores)))
        print('{} train samples, {} dev sample'.format(len(memory), len(dev_memory)))

    with benchmark('time_mdp') as time_mdp:
        mdpnet_msemu = MDPnet(config, 'mse_mu', 0)
        mdpnet_msepi_list = [MDPnet(config, 'mse_pi', i) for i in range(len(factual_types))]
        mdpnet_repbm_list = [MDPnet(config, 'repbm', i) for i in range(len(factual_types))]
        mdpnet_msemu.fit(memory, dev_memory)
        [mdpnet.fit(memory, dev_memory) for mdpnet in mdpnet_msepi_list + mdpnet_repbm_list]

    with benchmark('time_tc') as time_tc:
        tc = TerminalClassifier(config)
        tc.fit(memory, dev_memory)

    with benchmark('time_eval') as time_eval:
        print('Evaluate models using evaluation policy on the same initial states')
        mdpnet_msemu.eval()
        [mdpnet.eval() for mdpnet in mdpnet_msepi_list + mdpnet_repbm_list]
        target = np.zeros(config.sample_num_traj)

        init_states = []
        for i_episode in range(config.sample_num_traj):
            env.seed(seedvec[i_episode].item())
            init_states.append(preprocess_state(env.reset(), config.state_dim))
        init_states = torch.cat(init_states)
        mv_msemu = rollout_batch(init_states, mdpnet_msemu, tc, config.eval_num_rollout, egbehavior,
                                 epsilon=0, action_size=config.action_dim, maxlength=config.max_length, config=config)
        mv_msepi_list = []
        mv_repbm_list = []
        for mdpnet_msepi, mdpnet_repbm in zip(mdpnet_msepi_list, mdpnet_repbm_list):
            mv_msepi_list.append(rollout_batch(init_states, mdpnet_msepi, tc, config.eval_num_rollout, egbehavior,
                                               epsilon=0, action_size=config.action_dim, maxlength=config.max_length, config=config))
            mv_repbm_list.append(rollout_batch(init_states, mdpnet_repbm, tc, config.eval_num_rollout, egbehavior,
                                               epsilon=0, action_size=config.action_dim, maxlength=config.max_length, config=config))

    with benchmark('time_gt') as time_gt:
        for i_episode in range(config.sample_num_traj):
            values = deque()
            for i_trials in range(1):
                env.seed(seedvec[i_episode].item())
                state = preprocess_state(env.reset(), config.state_dim)
                true_state = state
                true_done = False
                true_steps = 0
                true_rewards = 0
                while not true_done:
                    true_action = eval_agent.act(true_state)
                    true_next_state, true_reward, true_done, _ = env.step(true_action)
                    true_state = preprocess_state(true_next_state, config.state_dim)
                    true_steps += 1
                    true_rewards += true_reward
                values.append(true_rewards)
            target[i_episode] = np.mean(values)

    print('| Sampling: {:.3f}s | Learn MDP: {:.3f}s | Learn TC: {:.3f}s | Eval MDP: {:.3f}s | Eval: {:.3f}s |'
          .format(time_sampling.time, time_mdp.time, time_tc.time, time_eval.time, time_gt.time))
    print('Target policy value:', np.mean(target))
    results = [mv_msemu] + mv_msepi_list + mv_repbm_list
    return results, target
