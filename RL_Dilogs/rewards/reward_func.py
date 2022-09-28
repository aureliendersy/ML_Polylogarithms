"""
Define the reward function that we want to use
"""

from abc import abstractmethod
from logging import getLogger
import sympy as sp

logger = getLogger()


class RewardFunction:
    """Core class"""
    def __init__(self, reward_on_end):
        self.reward_on_end = reward_on_end

    @abstractmethod
    def compute_reward(self, obs_rep, min_poly_hist, action_name, status_ep, goal_eq):
        raise NotImplementedError


class StepReward(RewardFunction):
    """
    Class for rewards given at each step
    """
    def __init__(self, len_pen, num_terms_pen, active_pen=True, simple_reward=None, end_penalty=0, pen_cycle=False):
        super(StepReward, self).__init__(False)

        self.length_penalty = len_pen
        self.num_terms_penalty = num_terms_pen
        self.active_penalty = active_pen
        self.simplification_reward = simple_reward
        self.end_penalty = end_penalty
        self.penalize_cycle = pen_cycle
        logger.info('We create a reward/penalty at each step')
        logger.info('Reward function has a penalty each step.'
                    ' Number of terms times -{}. We also penalize lengthy expressions with -{} per word/arg'.
                    format(str(self.num_terms_penalty), str(self.length_penalty)))
        if self.simplification_reward is not None:
            logger.info('Reward function also rewards each simplification with +{}'
                        .format(str(self.simplification_reward)))
        logger.info('Final expression not simplified are penalized with {}'.format(self.end_penalty))

    def compute_reward(self, obs_rep, min_poly_hist, action_name, status_ep, goal_eq=0):
        """For a given state and action pair compute the associated reward"""

        len_expr = obs_rep.get_length_expr()
        penalty_len = -self.length_penalty * len_expr

        end_penalty = self.end_penalty if status_ep else 0

        if self.active_penalty and action_name == 'cyclic':
            return penalty_len
        else:
            num_polylogs = obs_rep.count_number_polylogs()
            penalty_poly_num = -num_polylogs*self.num_terms_penalty

            if self.simplification_reward is not None and num_polylogs < min_poly_hist:
                return penalty_len + penalty_poly_num + self.simplification_reward + end_penalty
            else:
                return penalty_len + penalty_poly_num + end_penalty


class EndReward(RewardFunction):
    """
    Class for rewards given only at the end of the episode
    """
    def __init__(self, end_len_pen=0, num_terms_pen=0, end_match=1):
        super(EndReward, self).__init__(True)
        self.end_length_penalty = end_len_pen
        self.end_match = end_match
        self.penalize_cycle = False
        self.simplification_reward = None
        self.num_terms_penalty = num_terms_pen
        logger.info('We create a reward/penalty at the end of the episode')
        logger.info('Simplified expression get + {}'.format(str(end_match)))
        logger.info('Reward has a penalty only at the end. Number of terms times -{}.'
                    ' We also penalize long sequences with -{} per word/arg'.
                    format(str(self.num_terms_penalty), str(self.end_length_penalty)))

    def compute_reward(self, obs_rep, min_poly_hist, action_name, status_ep, goal_eq=0):
        """Get a reward/penalty only at the end"""
        if status_ep:
            len_expr = obs_rep.get_length_expr()
            penalty_len = -self.end_length_penalty * len_expr

            num_polylogs = obs_rep.count_number_polylogs()
            penalty_poly_num = -num_polylogs*self.num_terms_penalty
            if num_polylogs == 0:
                return self.end_match
            else:
                return penalty_len + penalty_poly_num
        else:
            return 0


class AdaptativeReward(RewardFunction):
    """
    Reward function that is adapted to the type of function coming in as input
    If we know the end goal we can try to steer the algorithm to match that end goal
    Relevant if we want to reach non zero goals
    """

    def __init__(self, len_pen, num_terms_pen, active_pen=True, simple_reward=None, end_penalty=0, pen_cycle=False):
        super(AdaptativeReward, self).__init__(False)
        self.length_penalty = len_pen
        self.num_terms_penalty = num_terms_pen
        self.active_penalty = active_pen
        self.simplification_reward = simple_reward
        self.penalize_cycle = pen_cycle
        self.end_penalty = end_penalty
        self.goal_eq = None
        logger.info('We create an adaptative reward/penalty at each step')
        logger.info('Reward function has a penalty each step.'
                    ' Number of terms times -{}. We also penalize lengthy expressions with -{} per word/arg'.
                    format(str(self.num_terms_penalty), str(self.length_penalty)))
        if self.simplification_reward is not None:
            logger.info('Reward function also rewards each simplification with +{}'
                        .format(str(self.simplification_reward)))

    def compute_reward(self, obs_rep, min_poly_hist, action_name, status_ep, goal_eq):
        """For a given state and action pair compute the associated reward
        We can now specify a given goal state"""

        len_goal = len(obs_rep.sympy_to_prefix(goal_eq))
        len_expr = obs_rep.get_length_expr()

        # Penalty is based on the distance to the goal
        penalty_len = -self.length_penalty * (len_expr - len_goal)
        end_penalty = self.end_penalty if status_ep else 0

        if self.active_penalty and action_name == 'cyclic':
            return penalty_len
        else:
            num_polylogs = obs_rep.count_number_polylogs()
            num_poly_goal = goal_eq.count(sp.polylog)
            penalty_poly_num = -(num_polylogs-num_poly_goal)*self.num_terms_penalty

            if self.simplification_reward is not None and num_polylogs < min_poly_hist:
                return penalty_len + penalty_poly_num + self.simplification_reward + end_penalty
            else:
                return penalty_len + penalty_poly_num + end_penalty
