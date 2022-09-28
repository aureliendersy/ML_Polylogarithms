"""
Contains all routines that are relevant for defining the action and observation spaces
"""

from abc import abstractmethod
from logging import getLogger
import numpy as np
import sympy as sp
from gym import spaces

from envs.utils_env import cyclic_perm_arg, get_inv_arg_poly, get_refl_arg_poly,\
    get_first_polylog_obs, get_dupli_arg_poly
from .utils_obs import BASE_SYMPY_OPERATORS, BASE_OPERATORS, VAR_LIST, SYMB_LIST, INC_ACTIONS
from .utils_obs import write_infix, parse_int, write_int

logger = getLogger()


class ActionSpace:

    def __init__(self, actions, multi_dim=False):
        self.actions = actions
        if not isinstance(actions, list):
            raise TypeError('Actions supplemented must be a list of action names')

        # Multi dim is not currently fully implemented
        if multi_dim:
            raise NotImplementedError
        self.multi_dim = multi_dim
        self.space = self.create_action_space()

        logger.info('Creating an Action space with {}'.format(str(self.actions)))
        logger.info('Action is multi-dimensional : {}'.format(self.multi_dim))
        logger.info('Action space is {}'.format(str(self.space)))

    def create_action_space(self):
        if self.multi_dim:
            if not isinstance(self.actions[0], list):
                raise TypeError('Actions supplied for multi_dim action space must be a multi-dim list of names')

            if not isinstance(self.actions[0][0], str):
                raise TypeError('Action list must have depth 2 only')

            dim_actions = [len(sublist) for sublist in self.actions]
            return spaces.MultiDiscrete(dim_actions)

        else:
            if not isinstance(self.actions[0], str):
                raise TypeError('Must have a single action list if not multi_dim space')
            return spaces.Discrete(len(self.actions))

    def get_action_space(self):
        """Return the actual action space"""
        return self.space

    def action_id_map(self):
        if not self.multi_dim:
            return {action: ids for ids, action in enumerate(self.actions)}
        else:
            dictret = {}
            for level, actions in enumerate(self.actions):
                dictret.update({action: [level, ids] for ids, action in enumerate(actions)})
            return dictret

    def id_action_map(self):
        dict_action_id = self.action_id_map()
        return {str(value): key for key, value in dict_action_id.items()}

    def get_action_name(self):
        """Return the name of all available actions"""
        if self.multi_dim:
            return [action for sub_action in self.actions for action in sub_action]
        else:
            return self.actions

    def __repr__(self):
        str_in = str(self.actions)
        return str_in + repr(self.get_action_space())


class ObservationSpace:

    def __init__(self, init_eq, dimension, req_feature_extractor, one_hot_encode):
        self.dim = dimension
        self.one_hot_encode = one_hot_encode
        self.init_eq = init_eq
        self.req_feature_extractor = req_feature_extractor
        self.min_poly_hist = None
        self.state = self.initialize_state()
        self.space = None
        logger.info('Create an Observation space of size {}'.format(self.dim))

    @abstractmethod
    def initialize_state(self):
        raise NotImplementedError

    def get_dim(self):
        return self.dim

    @abstractmethod
    def is_minimal(self, action_names, goal_state):
        """Given a list of possible action name, is the observation we have
        the minimal one. i.e can we simplify it further ?"""
        raise NotImplementedError

    @abstractmethod
    def count_number_polylogs(self):
        """How many polylogs are left in the current observation"""
        raise NotImplementedError

    def get_length_expr(self):
        """Depending on the observation space we can have different ways of defining what
        we mean by the length of an expression"""
        raise NotImplementedError


class StringObs(ObservationSpace):
    """
    Main class for hosting the observation space corresponding to a linear combination of dilogarithms
    """

    def __init__(self, init_eq, max_dim_str, sp_ops=BASE_SYMPY_OPERATORS, ops=BASE_OPERATORS, var=VAR_LIST,
                 symb_list=SYMB_LIST, one_hot_encode=False, simple_form='cancel', numeral_decomp=False,
                 reduced_graph=False, prev_st_info=False):
        if not isinstance(init_eq, sp.Expr):
            raise TypeError('Initial expression should be a sympy equation')

        self.sp_operators = sp_ops
        self.variables = var
        self.operators = ops

        self.simple_form = simple_form
        self.reduced_graph = reduced_graph
        self.numeral_decomp = numeral_decomp

        self.previous_state_info = prev_st_info

        self.symbols = symb_list
        self.elements = [str(i) for i in range(11)] if self.numeral_decomp else [str(i) for i in range(10)]
        self.words = ['pad'] + self.variables + list(self.operators.keys()) + self.symbols + self.elements
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}

        self.sp_state = None
        self.prefix_state = None
        super(StringObs, self).__init__(init_eq, max_dim_str, False, one_hot_encode)

        # If we one hot encode each word gets represented by a vector with the size of the alphabet
        if self.one_hot_encode:
            space = spaces.Box(low=0, high=1, shape=(self.dim, len(self.id2word)), dtype=np.int32)

        # In label encoding each word is just an integer token
        else:
            space = spaces.Box(low=0, high=len(self.words) - 1, shape=(self.dim,), dtype=np.int32)

        # If we include the information from the previous state we add (used to tune the reward)
        # 1) latest action taken; 2) latest number of polylogs; 3) Minimal number of polylogs throughout the trajectory
        if self.previous_state_info:
            self.space = spaces.Dict({'words': space,
                                      'prev_state': spaces.Box(low=np.array([-1, 0, 0]), high=np.array([3, 14, 14]),
                                                               shape=(3,), dtype=np.int32)})

        # If we don't need the extra information for the reward we can keep things as they are
        else:
            self.space = space
        if self.numeral_decomp:
            logger.info('We use a numeral decomposition for the integers')
        logger.info('We use {} variables'.format(str(self.variables)))
        logger.info('We use {} operators'.format(str(self.operators.keys())))
        logger.info('We use {} symbols'.format(str(self.symbols)))
        logger.info('We use {} elements'.format(str(self.elements)))
        logger.info('The full list of words is {}'.format(self.words))
        logger.info('The initial expression is {}'.format(self.init_eq))
        if self.reduced_graph:
            logger.info('We use a reduced graph representation')

    def initialize_state(self):
        """For the NLP observation we must translate our initial equation as a string"""
        self.sp_state = self.init_eq
        self.prefix_state = self.sympy_to_prefix(self.sp_state)
        self.min_poly_hist = self.count_number_polylogs()
        return self.prefix_to_obs(self.prefix_state)

    def _sympy_to_prefix(self, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        n_args = len(expr.args)

        assert (op == 'add' or op == 'mul') and (n_args >= 2) or (op != 'add' and op != 'mul') and (1 <= n_args <= 2)

        # square root
        if op == 'pow' and isinstance(expr.args[1], sp.Rational) and expr.args[1].p == 1 and expr.args[1].q == 2:
            return ['sqrt'] + self.sympy_to_prefix(expr.args[0])

        # parse children
        parse_list = []
        for i in range(n_args):
            if i == 0 or (i < n_args - 1):
                parse_list.append(op)
            parse_list += self.sympy_to_prefix(expr.args[i])

        return parse_list

    def sympy_to_prefix(self, expr):
        """
        Convert a SymPy expression to a prefix one.
        """
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return write_int(int(str(expr)), numeral_decomp=self.numeral_decomp)
        elif isinstance(expr, sp.Rational):
            return ['div'] + write_int(int(expr.p), numeral_decomp=self.numeral_decomp) +\
                   write_int(int(expr.q), numeral_decomp=self.numeral_decomp)
        elif expr == sp.E:
            return ['E']
        elif expr == sp.pi:
            return ['pi']
        elif expr == sp.I:
            return ['I']
        # SymPy operator
        for op_type, op_name in self.sp_operators.items():
            if isinstance(expr, op_type):
                return self._sympy_to_prefix(op_name, expr)
        # unknown operator
        raise TypeError(f"Unknown SymPy operator: {expr}")

    def _prefix_to_infix(self, expr):
        """
        Parse an expression in prefix mode, and output it in either:
          - infix mode (returns human readable string)
          - develop mode (returns a dictionary with the simplified expression)
        """
        if len(expr) == 0:
            raise ValueError("Empty prefix list.")
        t = expr[0]
        if t in self.operators:
            args = []
            l1 = expr[1:]
            for _ in range(self.operators[t]):
                i1, l1 = self._prefix_to_infix(l1)
                args.append(i1)
            return write_infix(t, args), l1
        elif t in self.variables or t == 'I':
            return t, expr[1:]
        else:
            val, i = parse_int(expr)
            return str(val), expr[i:]

    def prefix_to_sympy(self, expr):
        parsed, residual = self._prefix_to_infix(expr)
        if len(residual) > 0:
            raise NameError(f"Incorrect prefix expression \"{expr}\". \"{residual}\" was not parsed.")
        return sp.parse_expr(parsed, evaluate=False)

    def prefix_to_obs(self, prefix_expr):
        """
        Convert a prefix expression to a numpy array with each word encoded and zero padded
        :param prefix_expr:
        :return:
        """
        if not isinstance(prefix_expr, list):
            raise TypeError('I am waiting a prefix list here not {}'.format(prefix_expr))

        if len(prefix_expr) > self.dim:
            raise OverflowError('Dimension of prefix expression is too large.'
                                ' We can have {} words at most'.format(self.dim))
        else:
            # Pad with -1 which is not the coded value of any word
            coded_expr = [self.word2id[word] for word in prefix_expr]
            pad_index = self.word2id['pad']
            padded_expr = coded_expr + [pad_index]*(self.dim-len(prefix_expr))

            if self.one_hot_encode:
                onehot_encoded = list()
                for value in padded_expr:
                    letter = [0 for _ in range(len(self.word2id))]
                    if value != pad_index:
                        letter[value] = 1
                    onehot_encoded.append(letter)
                return np.array(onehot_encoded).astype(np.int32)
            else:
                return np.array(padded_expr).astype(np.int32)

    def obs_to_prefix(self, obs):
        """
        Convert an observation to a prefix expression
        :param obs:
        :return:
        """
        if self.one_hot_encode:
            non_zero = obs[np.any(obs != 0, axis=1)]
            prefix_expr = [self.id2word[np.where(vect != 0)[0][0]] for vect in non_zero]
        else:
            prefix_expr = [self.id2word[word] for word in obs[obs != 0]]
        return prefix_expr

    def return_add_token(self):
        """Return the token if corresponding to the add token"""
        if self.one_hot_encode:
            add_token = [0 for _ in range(len(self.word2id))]
            add_token[self.word2id['add']] = 1
        else:
            add_token = self.word2id['add']

        return add_token

    def obs_to_graph(self, obs):
        """
         Convert an observation to a graph with a list of nodes and directed edges
        :param obs:
        :return:
        """

        # Recover the prefix format
        prefix_expr = self.obs_to_prefix(obs)
        nodes = []
        edges = []
        parent = None
        bin_op_mem = []

        add_token = self.return_add_token()
        node_num = 0

        for i, token in enumerate(prefix_expr):

            # Get the node values
            if self.one_hot_encode:
                new_node = [0 for _ in range(len(self.word2id))]
                new_node[self.word2id[token]] = 1
            else:
                new_node = self.word2id[token]

            # Add a new branch to the tree. If we have a reduced graph representation then the tree is made wide
            # with respect to the overall addition token
            if parent is not None and nodes[parent] == add_token and new_node == add_token and self.reduced_graph:
                node_num -= 1
                bin_op_mem.append(parent)
            else:
                nodes.append(new_node)

                if parent is not None:
                    edges.append([parent, node_num])

                if token in list(self.operators.keys()):
                    if self.operators[token] > 1:
                        bin_op_mem.append(node_num)

                if token in self.variables + self.elements and len(prefix_expr)-1 != i:
                    if len(bin_op_mem) == 0:
                        raise ValueError('Error in tree {} for {} at position {}'.format(prefix_expr, token, i))
                    parent = bin_op_mem[-1]
                    bin_op_mem.pop()
                else:
                    parent = node_num
            node_num += 1

        if len(bin_op_mem) > 0:
            raise ValueError('Non empty list of remaining ops. I still have {}'.format(str(bin_op_mem)))

        return nodes, edges

    def extract_graph_first_term(self, nodes, edges):
        """
        Given a graph we want to extract the subtree that corresponds to the first term in the linear combination
        :param nodes:
        :param edges:
        :return:
        """

        add_token = self.return_add_token()

        if nodes[0] != add_token:
            return nodes, edges
        else:
            end_tree_lim = [edge[0] for edge in edges].index(0, 1)
            edges_set = edges[1:end_tree_lim]
            nodes_set = nodes[1:edges_set[-1][-1]+1]
            edges_set_shifted = [[edge[0]-1, edge[1]-1] for edge in edges_set]
            return nodes_set, edges_set_shifted

    def act_on_observation(self, action_name):
        """
        For a given action as input we have to do something to our current state
        :param action_name:
        :return:
        """
        init_state_sp = self.sp_state
        init_state_prefix = self.prefix_state

        if action_name == 'cyclic':
            if self.count_number_polylogs() > 1:
                self.sp_state = cyclic_perm_arg(self.sp_state)
            else:
                self.sp_state = self.sp_state
        elif action_name in ['inversion', 'reflection', 'duplication']:
            term_to_change = get_first_polylog_obs(self.sp_state)

            if action_name == 'inversion':
                new_term = -get_inv_arg_poly(term_to_change, simple_form=self.simple_form)
            elif action_name == 'reflection':
                new_term = -get_refl_arg_poly(term_to_change, simple_form=self.simple_form)
            elif action_name == 'duplication':
                new_term = get_dupli_arg_poly(term_to_change, simple_form=self.simple_form)
            else:
                raise NameError('I did not expect to see {} here'.format(action_name))

            if self.count_number_polylogs() > 1:
                new_args = list(self.sp_state.args)
                new_first_term = new_args[0].replace(term_to_change, new_term)
                if new_first_term.count(sp.polylog) > 1:
                    new_args = list(new_first_term.args) + new_args[1:]
                else:
                    new_args[0] = new_first_term
                non_eval_expr = sp.Add(*tuple(new_args), evaluate=False)
                eval_expr = sp.Add(*tuple(new_args), evaluate=True)
                self.sp_state = non_eval_expr if len(non_eval_expr.args) <= len(eval_expr.args) else eval_expr
            else:
                self.sp_state = self.sp_state.replace(term_to_change, new_term)

        else:
            raise NameError('Action {} is not implemented'.format(action_name))

        self.prefix_state = self.sympy_to_prefix(self.sp_state)

        if self.count_number_polylogs() < self.min_poly_hist:
            self.min_poly_hist = self.count_number_polylogs()

        # If too long we do nothing
        if len(self.prefix_state) > self.dim:
            self.sp_state = init_state_sp
            self.prefix_state = init_state_prefix

        self.state = self.prefix_to_obs(self.prefix_state)

    def is_minimal(self, action_names, goal_state=None):
        """Depending on which actions we are allowed to do
        we can be minimal with either 1 or 0 polylogs left"""
        num_polylogs = self.count_number_polylogs()
        if goal_state is None:
            return (num_polylogs == 0 or
                    (num_polylogs == 1 and not any(action in INC_ACTIONS for action in action_names)))
        else:
            return num_polylogs <= goal_state.count(sp.polylog)

    def count_number_polylogs(self):
        """We look at the sympy expression to make sure how many polylogs are left"""
        return self.sp_state.count(sp.polylog)

    def get_length_expr(self):
        """For an observation given by a string of words the length is the number of words
        in the prefix version of the expression"""
        return len(self.prefix_state)
