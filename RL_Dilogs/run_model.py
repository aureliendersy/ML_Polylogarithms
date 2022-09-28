"""
Script to run a model on a test set retaining only the top action prediction
Also outputs detailed statistics
"""

import sympy as sp
import numpy as np
from sympy import polylog
from observations.spaces import StringObs, ActionSpace
from envs.gym_env import PolyLogLinExpr
from sb3_contrib import TRPO
from model.monitor_utils import evaluate_model
from rewards.reward_func import StepReward

if __name__ == '__main__':

    # Episode length
    max_step = 50

    # Reward function to use
    reward_func = StepReward(len_pen=0.00, num_terms_pen=0.0, active_pen=False, simple_reward=1, pen_cycle=False)

    # Define the environment and spaces
    x = sp.Symbol('x')
    exprtest = polylog(2, x)
    obs_space = StringObs(exprtest, 512, one_hot_encode=True, numeral_decomp=True, prev_st_info=True,
                          reduced_graph=True)
    act_space = ActionSpace(['inversion', 'reflection', 'cyclic', 'duplication'])
    rng = np.random.RandomState(1)
    env = PolyLogLinExpr(max_step, reward_func, obs_space, act_space, rng)

    # Load the model
    rl_model = TRPO.load("path_to_model")

    # Load the test set
    start_set = 'path_to_test_set'

    # Evaluate the model and print the statistics
    evaluate_model(rl_model, env, initial_start_set=start_set, deterministic=True, num_runs=1, verbose=False,
                   random_agent=False, distrib_info=True)


