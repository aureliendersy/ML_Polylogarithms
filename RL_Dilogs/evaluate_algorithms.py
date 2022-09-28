"""
File to run and test the classical algorithm as well as the reinforcement learning models
"""

import numpy as np
import sympy as sp
from model.monitor_utils import search_algo_set
from sb3_contrib import TRPO
from rewards.reward_func import StepReward, EndReward
from observations.spaces import StringObs, ActionSpace
from envs.gym_env import PolyLogLinExpr

if __name__ == '__main__':

    start_set = 'test_set_path'

    # Define the reward and episode length
    reward_func = StepReward(len_pen=0.00, num_terms_pen=0.0, simple_reward=1, active_pen=False, pen_cycle=False)
    episode_length = 50

    # Define the observation and action spaces
    obs_space = StringObs(sp.polylog(2, sp.Symbol('x')), 512, one_hot_encode=True, numeral_decomp=True,
                          prev_st_info=True, reduced_graph=True)
    act_space = ActionSpace(['inversion', 'reflection', 'cyclic', 'duplication'])
    rng = np.random.RandomState(1)
    env = PolyLogLinExpr(episode_length, reward_func, obs_space, act_space, rng)

    # Load the model
    rl_model = TRPO.load("path_to_model")

    name = ['base', 'cyclic', 'base_gnn', 'cyclic_gnn']

    _ = search_algo_set(start_set, type_algo='beam_search', verbose=False, distrib_info=True,
                        model=rl_model, env=env, beam_size=3, breadth_size=2, gamma=0.9)

    _ = search_algo_set(start_set, type_algo='beam_search', verbose=False, distrib_info=True,
                        model=rl_model, env=env, beam_size=1, breadth_size=1, gamma=0.9)

    _ = search_algo_set(start_set, type_algo='classical', verbose=False, distrib_info=True,
                        tree_depth=10, beam_size=1, complex_ops=False)
