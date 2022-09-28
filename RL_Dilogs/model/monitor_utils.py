"""
Routines useful for monitoring the algorithm performance, plots and so on
"""

import os
import random
import glob

import pandas as pd
import numpy as np
import sympy as sp
from logging import getLogger
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from .classical_algorithm import DILOGS_IDS, reduce_polylog_expression, construct_polylog_expression, bfs,\
    import_eqs_set, smarter_bfs, beam_search_network

logger = getLogger()


dict_stats_algos = {'trpo': ['ep_len_mean', 'ep_rew_mean', 'explained_variance',
                             'kl_divergence_loss', 'policy_objective', 'value_loss'],
                    'ppo': ['ep_len_mean', 'ep_rew_mean', 'entropy_loss', 'clip_fraction', 'loss',
                            'explained_variance', 'approx_kl', 'policy_gradient_loss', 'value_loss'],
                    'dqn': ['ep_rew_mean', 'exploration_rate', 'ep_len_mean', 'loss']}


# Taken from stable baselines 3 examples
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        pass
        # Create folder if needed
        # if self.save_path is not None:
        #    os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    logger.info(f"Num timesteps: {self.num_timesteps}")
                    logger.info(f"Best mean reward: {self.best_mean_reward:.2f}"
                                f" - Last mean reward per episode: {mean_reward:.2f}")

            # New best model, you could save the agent here
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Example for saving best model
                if self.verbose > 0:
                    logger.info(f"Saving new best model to {self.save_path}.zip")
                self.model.save(self.save_path)
        return True


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, window_size=50, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param window_size: (int) size of the window used for moving average
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=window_size)
    # Truncate x
    x = x[len(x) - len(y):]

    _ = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig(log_folder+'rewards.pdf', bbox_inches='tight')
    plt.close()


def plot_all_results(log_folders, save_loc, window_size=50, title='Learning Curve'):
    """
    Same as the above but plotting all curves on the same graph
    :param log_folders:
    :param save_loc:
    :param window_size:
    :param title:
    :return:
    """
    _ = plt.figure(title)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")

    for exp_num, log_folder in enumerate(log_folders):

        x, y = ts2xy(load_results(log_folder), 'timesteps')
        y = moving_average(y, window=window_size)
        # Truncate x
        x = x[len(x) - len(y):]

        plt.plot(x, y, label='Exp {}'.format(str(exp_num)))

    plt.savefig(save_loc, bbox_inches='tight')
    plt.close()


def plot_detailed_results(log_files, save_dir,  algo_names, name_exp, window_size=10):
    """
    Plot all of the training curves for the data stored in the log folders
    :param log_files:
    :param save_dir:
    :param algo_names:
    :param name_exp:
    :param window_size:
    :return:
    """
    assert len(log_files) == len(name_exp)
    assert len(log_files) == len(algo_names)

    # Read the data frames from the log files
    data_frames, headers, rel_cols = [], [], []
    for i, file_name in enumerate(log_files):
        with open(file_name, "rt") as file_handler:
            first_line = file_handler.readline()
            header = [term.split('/')[1] for term in first_line[:-1].split(',')]
            data_frame = pd.read_csv(file_handler, index_col=None, header=0, names=header)
            data_frame['Name'] = name_exp[i]
        data_frames.append(data_frame)
        rel_cols.append(dict_stats_algos[algo_names[i]])

    full_cols = []
    for rel_col in rel_cols:
        full_cols = list(np.unique(full_cols + rel_col))

    # Plot all of the statistics
    for col in full_cols:
        title = 'Training curve : {}'.format(col.replace('_', ' '))
        plt.style.use('seaborn-whitegrid')
        _ = plt.figure(title)
        plt.xlabel('Number of Timesteps', fontsize=16)
        if col == 'ep_len_mean':
            plt.ylabel('Mean episode length', fontsize=16)
        elif col == 'ep_rew_mean':
            plt.ylabel('Mean reward per episode', fontsize=16)
        else:
            plt.ylabel(col.replace('_', ' '))

        # Do some averaging to smooth the plots based on the window size
        for i, data in enumerate(data_frames):
            if col in rel_cols[i]:
                x = data['total_timesteps']
                y = data[col]
                y = moving_average(y, window=window_size)
                x = x[len(x) - len(y):]
                plt.plot(x, y, label=name_exp[i])
                plt.legend(fontsize=14)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(save_dir+col+'.pdf', bbox_inches='tight')
        plt.close()


def evaluate_model(model, env, initial_start_set=None, deterministic=True, num_runs=1, verbose=False,
                   random_agent=False, non_solved_set=None, distrib_info=False):
    """
    Detailed evaluation run that outputs all of the relevant statistics.
    This runs by only selecting the best action of the model
    :param model: Model to test
    :param env: PolyLog environment
    :param initial_start_set: Set of equations that we should test on
    :param deterministic: Whether the policy is run in a deterministic fashion
    :param num_runs: If we run stochastically we can do it multiple times
    :param verbose:
    :param random_agent: If we want to swap out for a random agent
    :param non_solved_set: If we want to check on a subset of equations
    :param distrib_info: If we want the breakdown by input complexity (e.g number of scrambles)
    :return:
    """
    if initial_start_set is not None:

        # Load the set and initialize the stats
        scr_expr, simple_expr, eq_info_complex = import_eqs_set(initial_start_set, add_info=distrib_info)
        count_poly_train = []
        count_words_train = []
        count_poly_scr = []
        count_words_scr = []
        poly_extra_train = []
        words_extra_train = []
        reward_tots_train = []
        steps_taken = []
        steps_taken_non_cyclic = []
        steps_taken_distrib = []
        steps_taken_solved = []
        steps_taken_solved_non_cyclic = []
        steps_taken_solved_distrib = []
        num_eqs_consider = len(scr_expr)
        solved_num_poly = 0
        solved_num_words = 0
        num_cyclic_traj = 0
        non_solved_resolved = 0
        simple_len = {'0': 0, '1': 0, '2': 0, '3': 0}
        solved_simple_len = {'0': 0, '1': 0, '2': 0, '3': 0}
        solved_simple_len_min = {'0': 0, '1': 0, '2': 0, '3': 0}

        if distrib_info:
            unique_label = np.unique(np.array(eq_info_complex), axis=0)
            dict_valid_count = {'s' + str(label[0]) + 't' + str(label[1]): 0 for label in unique_label}
            dict_valid_count_min = {'s' + str(label[0]) + 't' + str(label[1]): 0 for label in unique_label}
            dict_steps_taken = {'s' + str(label[0]) + 't' + str(label[1]): 0 for label in unique_label}
            dict_eq_done = {'s' + str(label[0]) + 't' + str(label[1]): 0 for label in unique_label}
            dict_nodes = {'s' + str(label[0]) + 't' + str(label[1]): 0 for label in unique_label}

        if non_solved_set is not None:
            with open(non_solved_set, 'r') as fread:
                indicesstr = fread.readlines()
                indices = [int(ind) for ind in indicesstr]
            print('We have {} equations not solved by the classical algorithm'.format(len(indices)))

        # Loop on all the equations
        for i, scr_eq in enumerate(scr_expr):

            # Sanity check to see if all equations fit the model
            if len(env.obs_rep.sympy_to_prefix(scr_eq)) > env.obs_rep.dim:
                num_eqs_consider -= 1
                if distrib_info:
                    print('Equation:  s' + str(eq_info_complex[i][0]) + 't' + str(eq_info_complex[i][1]) +
                          ' does not fit')
                continue
            env.init_eq = scr_eq

            reward_tot = 0
            steps_done = 0
            cyclic_traj = False
            steps_done_non_cyclic = 0
            steps_done_distrib = {'0': 0, "1": 0, '2': 0, '3': 0}
            done = False
            obs = env.reset()
            min_poly = env.obs_rep.count_number_polylogs()
            if not verbose:
                if i % 10 == 0:
                    print('Doing expression {}'.format(str(i)))
            else:
                print('Doing expression {}'.format(str(i)))
                if non_solved_set is not None and i in indices:
                    print("Equation was not solved by the classical algorithm")

            # Do the episode
            while not done:
                # Random agent chooses randomly on the policy
                if random_agent:
                    action = random.randint(0, len(env.action_rep.actions)-1)
                    action_distr = model.policy.get_distribution(
                        model.policy.obs_to_tensor(obs)[0]).distribution.probs.detach().numpy()[0]

                else:
                    # Get the action distribution and apply the best one
                    action, _states = model.predict(obs, deterministic=deterministic)
                    action_distr = model.policy.get_distribution(
                        model.policy.obs_to_tensor(obs)[0]).distribution.probs.detach().numpy()[0]

                steps_done += 1
                steps_done_distrib[str(action)] += 1

                # Count the number of "active" actions
                if action != 2:
                    steps_done_non_cyclic += 1
                    if distrib_info:
                        dict_nodes['s' + str(eq_info_complex[i][0]) + 't' + str(eq_info_complex[i][1])] += 1
                if verbose:
                    print('Action taken is : {}'.format(env.action_rep.id_action_map()[str(action)]))
                    print('Distributions were {}'.format(str(str(action_distr))))

                obs, rewards, dones, info = env.step(action)

                # Check if the expression is smaller
                poly_num = env.obs_rep.count_number_polylogs()
                if poly_num < min_poly:
                    min_poly = poly_num

                # Check for cyclic trajectories
                if rewards < 0:
                    cyclic_traj = True
                reward_tot += rewards
                if verbose:
                    print('Total reward is {}'.format(reward_tot))
                    env.render()
                    print('Arguments are ordered as ')
                    print(env.obs_rep.sp_state.args)
                    print('\n')
                done = dones

            # Keep in memory the data for the episode run
            steps_taken.append(steps_done)
            steps_taken_non_cyclic.append(steps_done_non_cyclic)
            steps_taken_distrib.append(steps_done_distrib)
            count_poly_train.append(env.obs_rep.count_number_polylogs())
            count_words_train.append(env.obs_rep.get_length_expr())
            extra_poly = env.obs_rep.count_number_polylogs() - simple_expr[i].count(sp.polylog)
            poly_extra_train.append(extra_poly)
            count_poly_scr.append(scr_eq.count(sp.polylog))
            extra_words = env.obs_rep.get_length_expr() - len(env.obs_rep.sympy_to_prefix(simple_expr[i]))
            words_extra_train.append(extra_words)
            count_words_scr.append(len(env.obs_rep.sympy_to_prefix(scr_eq)))
            reward_tots_train.append(reward_tot)

            simple_len[str(simple_expr[i].count(sp.polylog))] += 1

            num_cyclic_traj += 1 if cyclic_traj else 0

            # If we solved the equation we take more statistics
            if extra_poly == 0:
                solved_num_poly += 1
                steps_taken_solved.append(steps_done)
                steps_taken_solved_non_cyclic.append(steps_done_non_cyclic)
                steps_taken_solved_distrib.append(steps_done_distrib)
                solved_simple_len[str(simple_expr[i].count(sp.polylog))] += 1
                if distrib_info:
                    dict_valid_count['s' + str(eq_info_complex[i][0]) + 't' + str(eq_info_complex[i][1])] += 1
                    dict_steps_taken['s' + str(eq_info_complex[i][0]) + 't' + str(eq_info_complex[i][1])] += steps_done_non_cyclic
                if non_solved_set is not None and i in indices:
                    print('Could solve Eq.{} not solved by classical algorithm'.format(i))
                    non_solved_resolved += 1

            # If we solved the equation at some point in the trajectory (relevant if end point is not 0)
            if min_poly <= simple_expr[i].count(sp.polylog):
                solved_simple_len_min[str(simple_expr[i].count(sp.polylog))] += 1
                if distrib_info:
                    dict_valid_count_min['s' + str(eq_info_complex[i][0]) + 't' + str(eq_info_complex[i][1])] += 1

            if extra_words <= 0:
                solved_num_words += 1

            if distrib_info:
                dict_eq_done['s' + str(eq_info_complex[i][0]) + 't' + str(eq_info_complex[i][1])] += 1

        # Make sure the stats account for any discarded input that is too long
        for key, value in solved_simple_len.items():
            if value > 0:
                solved_simple_len[key] = round(value / simple_len[key], 2)

        for key, value in solved_simple_len_min.items():
            if value > 0:
                solved_simple_len_min[key] = round(value / simple_len[key], 2)

        # Also normalize the distribution values
        if distrib_info:
            for key, value in dict_steps_taken.items():
                dict_steps_taken[key] = round(value/dict_valid_count[key], 2)
            for key, value in dict_valid_count.items():
                dict_valid_count[key] = round(value*100 / dict_eq_done[key], 2)
            for key, value in dict_valid_count_min.items():
                dict_valid_count_min[key] = round(value*100 / dict_eq_done[key], 2)
            for key, value in dict_nodes.items():
                dict_nodes[key] = value / dict_eq_done[key]

        # Output the entire statistics
        print('Mean Total reward:', np.array(reward_tots_train).mean())
        print('Mean Total Steps:', np.array(steps_taken).mean())
        print('Mean Total Non cyclic Steps:', np.array(steps_taken_non_cyclic).mean())
        print('Action distribution:',
              {key: sum(distr[key] for distr in steps_taken_distrib) for key in steps_taken_distrib[0].keys()})
        print('Mean Total Steps for solved:', np.array(steps_taken_solved).mean())
        print('Mean Total Non cyclic Steps for solved:', np.array(steps_taken_solved_non_cyclic).mean())
        print('Action distribution for solved:',
              {key: sum(distr[key] for distr in steps_taken_solved_distrib) for key in
               steps_taken_solved_distrib[0].keys()})
        print('Mean final number of polylogs:', np.array(count_poly_train).mean())
        print('Mean initial number of polylogs :', np.array(count_poly_scr).mean())
        print('Mean final number of polylogs extra:', np.array(poly_extra_train).mean())
        print('Mean final length of expression:', np.array(count_words_train).mean())
        print('Mean initial length of expression:', np.array(count_words_scr).mean())
        print('Mean final length of expression extra:', np.array(words_extra_train).mean())
        print('Found {} cyclic trajectories. So for {} % of the expressions:'.format(num_cyclic_traj, str(int(
            100 * num_cyclic_traj / num_eqs_consider))))
        print('Distribution for solved by initial length: {}'.format(solved_simple_len))
        print('Distribution for solved in trajectory by initial length: {}'.format(solved_simple_len_min))
        if distrib_info:
            print('Distribution for solved: {}'.format(dict_valid_count))
            print('Distribution for solved in trajectory: {}'.format(dict_valid_count_min))
            print('Distribution for solved (non cyclic) steps: {}'.format(dict_steps_taken))
            print('Distribution for nodes visited: {}'.format(dict_nodes))
        print('Solved expressions, num polylogs {} %'.format(str(int(100*solved_num_poly/num_eqs_consider))))
        print('Solved expressions, num words {} %'.format(str(int(100*solved_num_words / num_eqs_consider))))
        if non_solved_set is not None:
            print('Solved expressions not done by classical: {} %'.format(str(int(100*non_solved_resolved / len(indices)))))

    # If we don't start with a set to check then we just try to solve the current environment
    else:

        print('We start with')
        env.render()
        print('Arguments are ordered as ')
        print(env.obs_rep.sp_state.args)
        print('\n')

        count_poly = []
        count_words = []
        reward_tots = []
        resolved = 0

        for run in range(num_runs):
            reward_tot = 0
            done = False
            obs = env.reset()

            while not done:
                if random_agent:
                    action = random.randint(0, len(env.action_rep.actions)-1)
                else:
                    action, _states = model.predict(obs, deterministic=deterministic)
                if verbose:
                    print('Action taken is : {}'.format(env.action_rep.id_action_map()[str(action)]))
                    print('Distributions were {}'.format(str(str(model.policy.get_distribution(
                        model.policy.obs_to_tensor(obs)[0]).distribution.probs.detach().numpy()[0]))))
                obs, rewards, dones, info = env.step(action)
                reward_tot += rewards
                if verbose:
                    env.render()
                    print('Arguments are ordered as ')
                    print(env.obs_rep.sp_state.args)
                    print('\n')
                done = dones

            if env.obs_rep.count_number_polylogs() == 0:
                resolved += 1

            count_poly.append(env.obs_rep.count_number_polylogs())
            count_words.append(env.obs_rep.get_length_expr())
            reward_tots.append(reward_tot)

        print('Over {} run(s) we have '.format(num_runs))
        print('Mean Total reward:', np.array(reward_tots).mean())
        print('Mean final number of polylogs:', np.array(count_poly).mean())
        print('Mean final length of expression:', np.array(count_words).mean())
        print('Mean Resolution to 0 : {} %'.format(str(int(100*(resolved/num_runs)))))


def search_algo_set(initial_set, type_algo='classical', verbose=False, distrib_info=False, **kwargs):
    """
    Wrapper for applying a classical algorithm or a model and trying to solve the equations in a given set
    Also provides the required statistics.
    :param initial_set:
    :param type_algo:
    :param verbose:
    :param distrib_info:
    :param kwargs:
    :return:
    """
    # Check that we have the correct arguments
    if type_algo == 'classical':
        if any([argument not in list(kwargs.keys()) for argument in ['tree_depth', 'beam_size', 'complex_ops']]):
            raise NameError('Need to provide tree_depth, beam_size, complex_ops for the classical algorithm')
    elif type_algo == 'beam_search':
        if any([argument not in list(kwargs.keys()) for argument in ['model', 'env', 'beam_size', 'breadth_size',
                                                                     'gamma']]):
            raise NameError('Need to provide model, env, beam_size, breadth_size, gamma for the beam search')
    else:
        raise NotImplementedError

    # Load the equations from the training set
    print('On set : {}'.format(initial_set))
    initial_set_file = open(initial_set, "r")
    initial_eqs = initial_set_file.readlines()

    # If we look at statistics based on the way the input equation is scrambled
    if distrib_info:
        eq_info = [eqs for eqs in initial_eqs[0::4]]
        eq_info_complex = [(int(eq[eq.find(" scrambles") - 1]), int(eq[eq.find(" different terms") - 1])) for eq in
                           eq_info]

    # Convert to sympy expressions
    simple_expr = [sp.sympify(eqs[:-1].split(': ')[-1]) for eqs in initial_eqs[1::4]]
    scr_expr = [sp.sympify(eqs[:-1].split(': ')[-1]) for eqs in initial_eqs[2::4]]

    # Look at how we change the length of the various expressions
    count_poly_train, count_poly_scr, poly_extra_train = [], [], []

    # Look at how successful we are at solving + how many steps it takes
    num_ids, num_ids_solved, eqs_non_solved, nodes_visits_solved = [], [], [], []
    solved_num_poly = 0

    # If we are interested in in-depth stats about the scramble length
    if distrib_info:
        unique_label = np.unique(np.array(eq_info_complex), axis=0)
        dict_valid_count = {'s' + str(label[0]) + 't' + str(label[1]): 0 for label in unique_label}
        dict_steps_taken = {'s' + str(label[0]) + 't' + str(label[1]): 0 for label in unique_label}
        dict_nodes = {'s' + str(label[0]) + 't' + str(label[1]): 0 for label in unique_label}
        dict_nodes_solved = {'s' + str(label[0]) + 't' + str(label[1]): 0 for label in unique_label}
        dict_eqs = {'s' + str(label[0]) + 't' + str(label[1]): 0 for label in unique_label}

    # Loop on the set of equations
    for i, scr_eq in enumerate(scr_expr):

        if i % 10 == 0:
            print('Doing expression {}'.format(str(i)))

        # For the Modified Best First Search
        if type_algo == 'classical':
            reduced_args, identity_list, nodes_visited =\
                reduce_polylog_expression(scr_eq, DILOGS_IDS, **kwargs)
            reduced_expr = construct_polylog_expression(reduced_args)
            sol_len = len(identity_list)
            num_ids.append(sol_len)

        # For the RL driven beam search
        elif type_algo == 'beam_search':
            reduced_expr, sol_len, nodes_visited = beam_search_network(scr_eq, **kwargs)
            if reduced_expr is None:
                print('Too Long')
                continue
            num_ids.append(sol_len)
        else:
            raise ValueError

        # Sort the stats based on the scramble distance
        if distrib_info:
            dict_nodes['s' + str(eq_info_complex[i][0]) + 't' + str(eq_info_complex[i][1])] += nodes_visited
            dict_eqs['s' + str(eq_info_complex[i][0]) + 't' + str(eq_info_complex[i][1])] += 1

        # If we solved the equation we add this to our stats
        if reduced_expr == 0:
            count_poly_train.append(0)
            extra_poly = 0 - simple_expr[i].count(sp.polylog)
            num_ids_solved.append(sol_len)
            nodes_visits_solved.append(nodes_visited)
            if distrib_info:
                dict_valid_count['s' + str(eq_info_complex[i][0]) + 't' + str(eq_info_complex[i][1])] += 1
                dict_steps_taken['s' + str(eq_info_complex[i][0]) + 't' + str(eq_info_complex[i][1])] += sol_len
                dict_nodes_solved['s' + str(eq_info_complex[i][0]) + 't' + str(eq_info_complex[i][1])] += nodes_visited
            if verbose:
                print('Solved Eq {} in {} identities'.format(i, sol_len))

        # If we did not manage to solve the equation we still take some statistics
        else:
            count_poly_train.append(reduced_expr.count(sp.polylog))
            extra_poly = reduced_expr.count(sp.polylog) - simple_expr[i].count(sp.polylog)
            eqs_non_solved.append(i)
            if verbose:
                print('Did not solve Eq {} after {} identities'.format(i, sol_len))

        poly_extra_train.append(extra_poly)
        count_poly_scr.append(scr_eq.count(sp.polylog))

        if extra_poly == 0:
            solved_num_poly += 1

    # We normalize our final stats by the number of valid examples or solved examples when relevant
    if distrib_info:
        for key, value in dict_steps_taken.items():
            dict_steps_taken[key] = value/dict_valid_count[key]
        for key, value in dict_nodes.items():
            dict_nodes[key] = value/dict_eqs[key]
        for key, value in dict_nodes_solved.items():
            dict_nodes_solved[key] = value/dict_valid_count[key]

    # Simple print out of the final statistics
    print('Mean final number of polylogs:', np.array(count_poly_train).mean())
    print('Mean initial number of polylogs :', np.array(count_poly_scr).mean())
    print('Mean final number of polylogs extra:', np.array(poly_extra_train).mean())
    print('Mean number of identity used: {}'.format(np.mean(num_ids)))
    if distrib_info:
        print('Distribution for solved: {}'.format(dict_valid_count))
        print('Distribution for solved steps: {}'.format(dict_steps_taken))
        print('Distribution of nodes visited: {}'.format(dict_nodes))
        print('Distribution of nodes visited for solved: {}'.format(dict_nodes_solved))
    print('Mean number of identity used for solved: {}'.format(np.mean(num_ids_solved)))
    print('Mean number of nodes visited for solved: {}'.format(np.mean(nodes_visits_solved)))
    print('Solved expressions, num polylogs {} %'.format(str(int(100*solved_num_poly/len(scr_expr)))))

    return eqs_non_solved


def import_curriculum(curriculum_dir):
    """Return the ordered list of file (by name) from the relevant directory"""

    if curriculum_dir is not None:
        file_list = glob.glob(curriculum_dir+"*.txt")
        file_list.sort()
    else:
        file_list = None

    return file_list
