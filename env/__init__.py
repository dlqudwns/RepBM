
from gym.envs.registration import register

register(
    id='ContinuousCartPole-v0',
    entry_point='env.continuous_cartpole:ContinuousCartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)