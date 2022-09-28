"""
Core of the custom GYM environment used for dealing with linear combinations of Polylogs
"""
import random
from abc import abstractmethod
from gym import spaces
import gym
import sympy as sp
import numpy as np
from copy import deepcopy
from logging import getLogger
from .utils_env import generate_random_starting_eq, load_starting_set, load_curriculum, scr_null_state_v2, scr_null_state
from rewards.reward_func import AdaptativeReward, StepReward
from observations.spaces import ActionSpace, StringObs

logger = getLogger()


class BaseGymEnv(gym.Env):
    """
    Custom env that follows the implementation of Gym
    Overall Wrapper for the PolyLog Env
    """

    def __init__(self, max_steps, reward_function, obs_rep, action_rep, random_start_args, log_dir, rng,
                 start_set=None, gen='Standard'):
        super(BaseGymEnv, self).__init__()

        # Episode parameters
        self.max_steps = max_steps
        self.steps_done = 0
        self.total_steps_done = 0

        # Reward function parameters
        self.reward_function = reward_function
        self.adaptative = isinstance(self.reward_function, AdaptativeReward)
        self.goal_state = None

        # Spaces
        self.action_rep = action_rep
        self.action_space = self.action_rep.space
        self.obs_rep = obs_rep
        self.observation_space = self.obs_rep.space

        # Initialize the trajectory data and starting equation
        self.previous_trajectory = {'action': -1, 'num_poly': self.obs_rep.min_poly_hist}
        self.init_eq = obs_rep.init_eq

        # Random number generator
        self.rng = rng
        self.generator = gen

        self.check_env()

        # Parameters for on the fly equation generation (as opposed to reading from a training set)
        self.random_start = random_start_args['random_start'] if random_start_args is not None else None
        self.start_size = random_start_args['start_size'] if random_start_args is not None else None
        self.max_len_simple = random_start_args['len_simple'] if random_start_args is not None else None
        self.max_len_scr = random_start_args['len_scr'] if random_start_args is not None else None
        self.max_scr_num = random_start_args['num_scr'] if random_start_args is not None else None

        # To read from a data file or a curriculum
        self.file_starts = random_start_args['file_starts'] if random_start_args is not None else None
        self.curriculum = random_start_args['curriculum'] if random_start_args is not None else None

        # Logger for random equation generation
        if self.random_start and not self.file_starts:
            logger.info('Starting from {} random multiple points'.format(str(self.start_size)))
            logger.info('Max length of simple expression : {}'.format(str(self.max_len_simple)))
            logger.info('Indicative length of scrambled expression : {}'.format(str(self.max_len_scr)))
            logger.info('Maximum number of scrambles per term : {}'.format(str(self.max_scr_num)))

        self.log_dir = log_dir

        # Construct the starting set
        self.start_set = self.create_starting_set(start_set)
        self.current_set = 1

        # If we generate a new starting set we save it
        if self.curriculum is None and start_set is None:
            self.save_start_set()

    @abstractmethod
    def step(self, action):
        """Take a step given an action as input"""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Reset the observation and returns it"""
        raise NotImplementedError

    @abstractmethod
    def render(self, mode="human"):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    def get_action_space(self):
        """Returns the action space that we are using"""
        return self.action_space

    def get_obs_dim(self):
        return self.obs_rep.get_dim()

    @abstractmethod
    def create_starting_set(self, start_set=None):
        raise NotImplementedError

    def check_env(self):
        """Sanity checks"""
        if self.reward_function.simplification_reward is not None or\
                self.reward_function.penalize_cycle:
            logger.info("We have an environment that requires previous state information for the reward")
            if not isinstance(self.obs_rep.space, spaces.Dict):
                raise AttributeError("We need to have a dictionary observation space")
            else:
                logger.info("We have a dictionary observation space")

    def save_start_set(self):
        """
        If we have some fixed starting state we save it in a file for reference
        :return:
        """
        if self.start_set is not None:
            if self.log_dir is None:
                raise ValueError('Need a path to store the set of starting points')

            # Generate also a Mathematica compatible output
            f = open(self.log_dir+"starts.txt", "a")
            f2 = open(self.log_dir+"starts_mma.txt", "a")
            for i, simple_eq in enumerate(self.start_set[1]):

                f.write("Example {}".format(str(i + 1)) + '\n')
                f.write('Simple expression : {}'.format(str(simple_eq)) + '\n')
                f.write('Scrambled expression : {}'.format(str(self.start_set[0][i])) + '\n')
                f.write('\n')

                f2.write("Example {}".format(str(i + 1)) + '\n')
                f2.write('Simple expression : {}'.format(str(sp.mathematica_code(simple_eq))) + '\n')
                f2.write('Scrambled expression : {}'.format(str(sp.mathematica_code(self.start_set[0][i]))) + '\n')
                f2.write('\n')


class PolyLogLinExpr(BaseGymEnv):
    """Environment used to represent any linear combination of polylogarithms"""

    def __init__(self, max_steps, reward_function, obs_rep, action_rep, rng, random_start_args=None, log_dir=None,
                 start_set=None, gen='Standard'):
        super(PolyLogLinExpr, self).__init__(max_steps, reward_function, obs_rep, action_rep, random_start_args,
                                             log_dir, rng, start_set=start_set, gen=gen)

    def step(self, action):
        """Given a particular action we have to do something to our expression
        We update the state, calculate the rewards, update the number of steps done
        and check if we reached a terminal state"""

        # Retain the action that we have to perform
        action_name = self.action_rep.id_action_map()[str(action)]

        # Get information about the past action and minimal number of polylogs
        prev_action = self.previous_trajectory['action']
        prev_num_poly = self.previous_trajectory['num_poly']
        previous_min_poly = self.obs_rep.min_poly_hist

        # Perform the action on the given observation
        self.obs_rep.act_on_observation(action_name)
        self.steps_done += 1
        self.total_steps_done += 1

        # Goal state will actually be 0 in most/all cases
        goal_state = self.goal_state if self.adaptative else None

        # Did we reach a terminal state
        done = self.steps_done >= self.max_steps or self.obs_rep.is_minimal(self.action_rep.get_action_name(),
                                                                            goal_state=goal_state)

        # Compute the reward associated with reaching this state
        reward = self.reward_function.compute_reward(self.obs_rep, previous_min_poly, action_name, done,
                                                     goal_eq=goal_state)

        # Optional penalty to stop things from looping
        if prev_action == action and self.reward_function.penalize_cycle:
            if action_name in ['reflection', 'inversion'] and prev_num_poly == self.obs_rep.count_number_polylogs():
                reward -= 0.25
            # Also penalize cyclic duplications. Do this when the duplication increases the length,
            # In other words we allow for cyclic trajectories if they bring us back to some smaller expression
            elif action_name == 'duplication' and self.obs_rep.count_number_polylogs() >= prev_num_poly:
                reward -= 0.25

        # Update the information about the action taken and the number of polylogs
        self.previous_trajectory = {'action': action, 'num_poly': self.obs_rep.count_number_polylogs()}

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        # Return the observed state
        if not isinstance(self.obs_rep.space, spaces.Dict):
            state_return = self.obs_rep.state

        # For a dictionary type of observation we have to return both the both components
        else:
            prev_state = [action, self.obs_rep.count_number_polylogs(), self.obs_rep.min_poly_hist]
            state_return = {'words': self.obs_rep.state, 'prev_state': np.array(prev_state).astype(np.int32)}
        return state_return, reward, done, info

    def reset(self):
        """Reset the environment to the initial linear combination"""

        # If we have a new start
        if self.random_start:

            # If we need to generate a new starting point on the fly
            if self.start_set is None and self.curriculum is None:
                new_init_eq = None
                while new_init_eq is None or len(self.obs_rep.sympy_to_prefix(new_init_eq)) > self.obs_rep.dim * 0.75:
                    variable = sp.Symbol(self.obs_rep.variables[0])
                    act_gen = deepcopy(self.action_rep.actions)
                    act_gen.remove('cyclic')
                    if self.generator == 'Standard':
                        new_init_eq, _ = generate_random_starting_eq(self.max_len_simple, self.max_len_scr,
                                                                     self.max_scr_num, 2, 2, variable, 0.5, act_gen)
                    elif self.generator in ['V1', 'V2']:
                        num_scr = random.randint(1, self.max_scr_num)
                        num_add_zero = random.randint(1, min(self.max_len_scr, num_scr))

                        while new_init_eq is None or new_init_eq == 0:
                            if self.generator == 'V1':
                                new_init_eq = scr_null_state(num_scr, num_add_zero, 2, 2, variable, act_gen, verbose=False)
                            else:
                                new_init_eq = scr_null_state_v2(num_scr, num_add_zero, 2, 2, variable, act_gen,
                                                                verbose=False)
                # Stats about the new starting points generated on the fly
                if self.generator in ['V1', 'V2']:
                    f = open(self.log_dir + "starts_distrib.txt", "a")
                    f.write("Using {} scrambles and {} zeros".format(num_scr, num_add_zero) + '\n')
                    f.close()
                # Assign the new starting point
                self.obs_rep.init_eq = new_init_eq

            # If we are reading our new starting points from a training set
            else:
                new_init_eq = None

                # No curriculum so just read from the unique set
                if self.curriculum is None:
                    scr_eqs = self.start_set[0]
                    simple_eqs = self.start_set[1]

                # If curriculum need to figure out which set we are reading from
                else:
                    len_starts = [len(start_set) for start_set in self.start_set[0]]
                    index_start = int(self.total_steps_done/self.max_steps)
                    if index_start > 5*sum(len_starts[:self.current_set]):
                        self.current_set += 1
                        logger.info('Starting to use equations from {}'.format(self.curriculum[self.current_set-1]))
                    scr_eqs = self.start_set[0][self.current_set-1]
                    simple_eqs = self.start_set[1][self.current_set-1]

                # Take a random equation from that starting set and check that it is not too long
                while new_init_eq is None or len(self.obs_rep.sympy_to_prefix(new_init_eq)) > self.obs_rep.dim * 0.75:
                    random_index = self.rng.choice(range(len(scr_eqs)))
                    new_init_eq = scr_eqs[random_index]
                    self.goal_state = simple_eqs[random_index]
                self.obs_rep.init_eq = new_init_eq

        # If we always want to start from the same equation (e.g in debug mode)
        else:
            self.obs_rep.init_eq = self.init_eq

        # Initialize the environment at this new equation
        self.obs_rep.state = self.obs_rep.initialize_state()
        self.previous_trajectory = {'action': -1, 'num_poly': self.obs_rep.min_poly_hist}
        self.steps_done = 0

        # If we do not need the extra information
        if not isinstance(self.obs_rep.space, spaces.Dict):
            return self.obs_rep.state

        # If we need the info on the previous action and number of dilogs
        else:
            prev_state = [-1, self.obs_rep.min_poly_hist, self.obs_rep.min_poly_hist]
            return {'words': self.obs_rep.state, 'prev_state': np.array(prev_state).astype(np.int32)}

    def close(self):
        pass

    def render(self, mode="human"):
        """Print the relevant information"""
        print(self.obs_rep.sp_state)

    def create_starting_set(self, start_set=None):
        """
        Create a fix set of starting equations that we can draw from rather than
        generating a new one at each reset
        :return:
        """

        # If we want to generate the examples at each reset
        if self.start_size is None and start_set is None and self.file_starts is None and self.curriculum is None:
            return None

        # If we pass a set as input
        elif start_set is not None:
            logger.info("Reading given starting set")
            return start_set

        # If we need to generate the set
        elif isinstance(self.start_size, int) and self.file_starts is None and self.curriculum is None:
            start_set_scr = []
            start_set_simple = []
            variable = sp.Symbol(self.obs_rep.variables[0])
            act_gen = deepcopy(self.action_rep.actions)
            act_gen.remove('cyclic')

            # Generate the equations one by one
            for i in range(self.start_size):
                print('Generating eq {}'.format(i))
                valid_expr = False
                scr_eq, simple_eq = 0, 0
                while not valid_expr:

                    if self.generator == 'Standard':
                        scr_eq, simple_eq = generate_random_starting_eq(self.max_len_simple, self.max_len_scr,
                                                                        self.max_scr_num, 2, 2, variable, 0.5, act_gen)
                    elif self.generator in ['V1', 'V2']:
                        num_scr = random.randint(1, self.max_scr_num)
                        num_add_zero = random.randint(1, min(self.max_len_scr, num_scr))

                        # Generator V2 is preferred
                        if self.generator == 'V1':
                            scr_eq = scr_null_state(num_scr, num_add_zero, 2, 2, variable, act_gen, verbose=False)
                        else:
                            scr_eq = scr_null_state_v2(num_scr, num_add_zero, 2, 2, variable, act_gen, verbose=False)

                    valid_expr = len(self.obs_rep.sympy_to_prefix(scr_eq)) < (0.75 * self.obs_rep.dim)

                start_set_scr.append(scr_eq)
                start_set_simple.append(simple_eq)

            return [start_set_scr, start_set_simple]

        # If we pass the path to a set as input, then we load it
        elif isinstance(self.file_starts, str) and self.curriculum is None:
            return load_starting_set(self.file_starts, logger)

        # If we pass the path to curriculum sets as input, then we load them
        elif self.curriculum is not None:
            return load_curriculum(self.curriculum, logger)

        else:
            return None

#
# class FrontPolyLog(PolyLogLinExpr):
#     """
#     Front end if we have a parser
#     """
#     def __init__(self, max_steps=50, rw_cls='StepReward', len_pen=0, num_terms_pen=0, active_pen=False, simple_reward=1,
#                  pen_cycle=True, obs_cls='StringObs', exprtest=sp.polylog(2, sp.Symbol('x')), dimobs=512, one_hot=True,
#                  numeral_decomp=True, reduced_g=False, prev_st_info=True, act_cls='ActionSpace', act_list='4_actions',
#                  rng=0, log_d='', random_start=True, file_starts='', curriculum=None):
#
#         if act_cls == 'ActionSpace' and act_list == '4_actions':
#             act_space = ActionSpace(['inversion', 'reflection', 'cyclic', 'duplication'])
#         else:
#             raise NotImplementedError
#
#         if obs_cls == 'StringObs':
#             obs_space = StringObs(exprtest, dimobs, one_hot_encode=one_hot, numeral_decomp=numeral_decomp,
#                                   reduced_graph=reduced_g, prev_st_info=prev_st_info)
#         else:
#             raise NotImplementedError
#
#         if rw_cls == 'StepReward':
#             reward_func = StepReward(len_pen=len_pen, num_terms_pen=num_terms_pen, active_pen=active_pen,
#                                      simple_reward=simple_reward, pen_cycle=pen_cycle)
#         elif rw_cls == 'AdaptativeReward':
#             reward_func = AdaptativeReward(len_pen=len_pen, num_terms_pen=num_terms_pen, active_pen=active_pen,
#                                            simple_reward=simple_reward, pen_cycle=pen_cycle)
#         else:
#             raise NotImplementedError
#
#         rng = np.random.RandomState(rng)
#         random_starts = {'random_start': random_start, 'file_starts': file_starts,
#                          'curriculum': curriculum, 'start_size': 100, 'len_simple': 0, 'len_scr': 2, 'num_scr': 0}
#
#         super(FrontPolyLog, self).__init__(max_steps, reward_func, obs_space, act_space, rng,
#                                            random_start_args=random_starts, log_dir=log_d)
#
