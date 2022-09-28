"""
Main script to run for training the RL models
"""

import os
import time
import random
from logger import create_logger
from rewards.reward_func import StepReward
import sympy as sp
import numpy as np
from observations.spaces import StringObs, ActionSpace
from envs.gym_env import PolyLogLinExpr
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from model.rl_algorithms import TRPOModel
from model.monitor_utils import SaveOnBestTrainingRewardCallback, plot_results, plot_all_results, import_curriculum

if __name__ == '__main__':

    logger = create_logger('run.log')
    logger.info("============ Initialized logger ============")

    # Define the training timesteps along with the episode length
    max_step = 50
    total_t_steps = 3000000
    check_freq = 30000

    # Reward function to use
    reward_func = StepReward(len_pen=0.00, num_terms_pen=0.0, active_pen=False, simple_reward=1, pen_cycle=False)

    # Use a sympy observation environment
    x = sp.Symbol('x')
    expr0 = 2 * sp.polylog(2, x)
    obs_space = StringObs(expr0, 512, one_hot_encode=True, numeral_decomp=True, reduced_graph=True, prev_st_info=True)

    # Use an Discrete action space with 3 actions
    act_space = ActionSpace(['inversion', 'reflection', 'cyclic', 'duplication'])

    # At each new episode we get a new equation
    random_start = True

    # If we want to do curriculum learning
    curriculum_dir = None

    # Define the parameters for the data generation if we desire to do it on the fly
    start_size = 13500  # Training set size
    len_simple = 0
    len_scr = 4
    num_scr = 7

    # Otherwise we can load a previously generated training set
    file_starts = 'training_set_path'

    random_start_args = {'random_start': random_start, 'start_size': start_size, 'len_simple': len_simple,
                         'len_scr': len_scr, 'num_scr': num_scr, 'file_starts': file_starts,
                         'curriculum': import_curriculum(curriculum_dir)}

    # Directory to store result
    log_dir = "log_dir"

    # RL Algorithm

    # Network architecture
    policy = 'MultiInputPolicy'
    shared_layers = [256]
    vf_layers = [128, 128, 64]
    pi_layers = [128, 128, 64]
    act_func = 'relu'

    # Agent parameters
    model_name = 'trpo'
    trpo_kwargs = {'gae_lambda': 0.9, 'gamma': 0.9, 'n_steps': 2048}
    ft_extrac = {'name': 'sage', 'params': {'embed_dim': 64, 'num_layers': 2, 'obs_space': obs_space,
                                            'bidirectional': True}}
    # ft_extrac = 'Flatten'

    num_experiments = 1
    times = []
    log_dirs = []

    # Can repeat experiment with different seeds
    for experiment in range(num_experiments):
        np.random.seed(experiment)
        random.seed(experiment)
        rng = np.random.RandomState(experiment)

        # Path to store results
        log_link = log_dir + model_name + '_exp_{}'.format(str(experiment)) + '/'
        log_dirs.append(log_link)
        os.makedirs(log_link, exist_ok=True)

        # Define the environment and check it
        env = Monitor(PolyLogLinExpr(max_step, reward_func, obs_space, act_space, rng,
                                     random_start_args=random_start_args, log_dir=log_link, gen='V2'), log_link)
        env.reset()

        # Define the agent
        base_model = TRPOModel(env, policy, shared_layers, vf_layers, pi_layers, act_func, feature_extractor=ft_extrac,
                               seed=experiment, one_hot=obs_space.one_hot_encode, **trpo_kwargs)

        # Configure the logger
        assert base_model.get_name() == model_name
        rl_model = base_model.get_model()
        logger.info(str(rl_model.policy))
        new_logger = configure(log_link, ["stdout", "csv"])
        rl_model.set_logger(new_logger)

        # Start the timer
        start = time.time()

        # Use deterministic actions for evaluation
        eval_callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_link)

        # Now start the training
        rl_model.learn(total_timesteps=total_t_steps, callback=eval_callback)

        times.append(time.time() - start)

        # Save the final model and associated stats
        rl_model.save(log_link + 'final_model')
        stats_path = os.path.join(log_link, "vec_normalize_" + str(experiment) + ".pkl")

        # Plot the rewards as a function of time steps
        plot_results(log_link, window_size=int(check_freq/20))

    plot_all_results(log_dirs, log_dir + model_name + '_rewards.pdf', window_size=int(check_freq/20))
