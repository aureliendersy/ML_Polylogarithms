# Adapted from https://github.com/facebookresearch/SymbolicMathematics under CC BY-NC 4.0

"""
Environment used to deal with the tokenization and generation of polylogarithmic expressions
"""

from logging import getLogger
import os
import io
import re
import sys
import itertools
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.cache import clear_cache
from sympy.calculus.util import AccumBounds
from ..utils import bool_flag
from ..utils import timeout, TimeoutError
from .sympy_utils import remove_overall_const_log
from .sympy_utils import simplify
from .sympy_utils import gen_proba_transcendental

CLEAR_SYMPY_CACHE_FREQ = 10000


SPECIAL_WORDS = ['<s>', '</s>', '<pad>', '(', ')']

EVAL_VALUES = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 2.1, 3.1]
EVAL_VALUES = EVAL_VALUES + [-x for x in EVAL_VALUES]

TEST_ZERO_VALUES = [0.1, 0.9, 1.1, 1.9]
TEST_ZERO_VALUES = [-x for x in TEST_ZERO_VALUES] + TEST_ZERO_VALUES
ZERO_THRESHOLD = 1e-13


logger = getLogger()


class ValueErrorExpression(Exception):
    pass


class UnknownSymPyOperator(Exception):
    pass


class InvalidPrefixExpression(Exception):

    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)


class CharSPEnvironment(object):

    TRAINING_TASKS = {'symbol_int', 'func_simple'}

    SYMPY_OPERATORS = {
        # Elementary functions. We consider only functions with rational arguments. Keep exp for future applications.
        sp.Add: 'add',
        sp.Mul: 'mul',
        sp.Pow: 'pow',
        sp.exp: 'exp',
        sp.log: 'ln',
        sp.Abs: 'abs',

        # Polylogs
        sp.polylog: 'polylog',
    }

    OPERATORS = {
        # Elementary functions
        'add': 2,
        'sub': 2,
        'mul': 2,
        'div': 2,
        'pow': 2,
        'inv': 1,
        'pow2': 1,
        'pow3': 1,
        'pow4': 1,
        'sqrt': 1,
        'exp': 1,
        'ln': 1,
        'abs': 1,

        # Symbol operator. By default use 2 arguments (weight 2) but will be overwritten if necessary.
        'citi': 2,

        # Polylogs. 2 Arguments: the transcendental weight and the functional arguments -> Li_n(x)
        'polylog': 2,
        'f': 1,
    }

    def __init__(self, params):

        self.session = None

        # MMA relevant parameters
        self.lib_path = params.lib_path
        self.kernel_path = params.kernel_path

        # Verification mode for the final result
        self.numerical_check = params.numerical_check
        self.symbol_check = params.symbol_check

        # Parameters relevant to the generation of polylogarithmic expressions
        self.max_int = params.max_int
        self.max_int_tot = params.max_int_tot
        self.max_ops = params.max_ops
        self.max_terms_transc = params.max_terms_transc
        self.max_ops_G = params.max_ops_G
        self.int_base = params.int_base
        self.balanced = params.balanced
        self.positive = params.positive
        self.precision = params.precision
        self.n_variables = params.n_variables
        self.n_coefficients = params.n_coefficients
        self.numeral_decomp = params.numeral_decomp
        self.max_len = params.max_len
        assert self.max_int >= 1
        assert abs(self.int_base) >= 2
        assert self.precision >= 2

        # Overwrite the number of arguments of the symbol operator (with the transcendental weight)
        self.OPERATORS['citi'] = params.task_arg

        # Parse operators with their weights
        self.operators = sorted(list(self.OPERATORS.keys()))
        ops = params.operators.split(',')
        ops = sorted([x.split(':') for x in ops])
        assert len(ops) >= 1 and all(o in self.OPERATORS for o, _ in ops)
        self.all_ops = [o for o, _ in ops]
        self.una_ops = [o for o, _ in ops if self.OPERATORS[o] == 1]
        self.bin_ops = [o for o, _ in ops if self.OPERATORS[o] == 2]
        logger.info(f"Unary operators: {self.una_ops}")
        logger.info(f"Binary operators: {self.bin_ops}")
        self.all_ops_probs = np.array([float(w) for _, w in ops]).astype(np.float64)
        self.una_ops_probs = np.array([float(w) for o, w in ops if self.OPERATORS[o] == 1]).astype(np.float64)
        self.bin_ops_probs = np.array([float(w) for o, w in ops if self.OPERATORS[o] == 2]).astype(np.float64)
        self.all_ops_probs = self.all_ops_probs / self.all_ops_probs.sum()
        self.una_ops_probs = self.una_ops_probs / self.una_ops_probs.sum()
        self.bin_ops_probs = self.bin_ops_probs / self.bin_ops_probs.sum()

        assert len(self.all_ops) == len(set(self.all_ops)) >= 1
        assert set(self.all_ops).issubset(set(self.operators))
        assert len(self.all_ops) == len(self.una_ops) + len(self.bin_ops)

        # Define the constants and variables + symbols for integers
        self.constants = ['pi']
        self.variables = OrderedDict({
            'x': sp.Symbol('x', real=True, nonzero=True),
        })
        self.symbols = ['I', 'INT+', 'INT-']

        # Integer representation
        if self.balanced:
            assert self.int_base > 2
            max_digit = (self.int_base + 1) // 2
            self.elements = [str(i) for i in range(max_digit - abs(self.int_base), max_digit)]
        else:
            # We add the number 10 in the dictionary to allow the numeral decomposition in base 10
            self.elements = [str(i) for i in range(abs(self.int_base)+1)]
        assert 1 <= self.n_variables <= len(self.variables)
        assert all(v in self.OPERATORS for v in self.SYMPY_OPERATORS.values())

        # SymPy local dictionary for parsing expressions
        self.local_dict = {}
        for k, v in list(self.variables.items()):
            assert k not in self.local_dict
            self.local_dict[k] = v

        # Words in the vocabulary
        self.words = SPECIAL_WORDS + self.constants + list(self.variables.keys()) + self.operators + self.symbols + self.elements
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        logger.info(f"words: {self.word2id}")

        # leaf probabilities
        s = [float(x) for x in params.leaf_probs.split(',')]
        assert len(s) == 4 and all(x >= 0 for x in s)
        self.leaf_probs = np.array(s).astype(np.float64)
        self.leaf_probs = self.leaf_probs / self.leaf_probs.sum()
        assert self.leaf_probs[0] > 0
        assert (self.leaf_probs[1] == 0) == (self.n_coefficients == 0)

        # Transcendental probabilities (general)
        if params.tasks == 'symbol_int' or 'symbol_int' in params.tasks:
            self.partitions, self.transc_probs = gen_proba_transcendental(params.task_arg, params.skew_basis_proba)

        # Fraction probabilities for the factors
        s3 = [float(x) for x in params.frac_probs.split(',')]
        assert len(s3) == 3 and all(x >= 0 for x in s3)
        self.frac_probs = np.array(s3).astype(np.float64)
        self.frac_probs = self.frac_probs / self.frac_probs.sum()

        # possible leaves
        self.n_leaves = self.n_variables + self.n_coefficients
        if self.leaf_probs[2] > 0:
            self.n_leaves += self.max_int * (1 if self.positive else 2)
        if self.leaf_probs[3] > 0:
            self.n_leaves += len(self.constants)
        logger.info(f"{self.n_leaves} possible leaves.")

        # generation parameters
        self.nl = 1  # self.n_leaves
        self.p1 = 1  # len(self.una_ops)
        self.p2 = 1  # len(self.bin_ops)

        # initialize distribution for binary and unary-binary trees
        self.bin_dist = self.generate_bin_dist(params.max_ops)
        self.ubi_dist = self.generate_ubi_dist(params.max_ops)

        # rewrite expressions
        self.rewrite_functions = [x for x in params.rewrite_functions.split(',') if x != '']
        assert len(self.rewrite_functions) == len(set(self.rewrite_functions))
        assert all(x in ['expand', 'factor', 'expand_log', 'logcombine', 'powsimp', 'simplify'] for x in self.rewrite_functions)

        # valid check
        if self.numerical_check:
            logger.info(f"Checking expressions in {str(EVAL_VALUES)}")

    def batch_sequences(self, sequences):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)
        assert lengths.min().item() > 2

        sent[0] = self.eos_index
        for i, s in enumerate(sequences):
            sent[1:lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    def generate_bin_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(0, n) = 0
            D(1, n) = C_n (n-th Catalan number)
            D(e, n) = D(e - 1, n + 1) - D(e - 2, n + 1)
        """
        # initialize Catalan numbers
        catalans = [1]
        for i in range(1, 2 * max_ops + 1):
            catalans.append((4 * i - 2) * catalans[i - 1] // (i + 1))

        # enumerate possible trees
        D = []
        for e in range(max_ops + 1):  # number of empty nodes
            s = []
            for n in range(2 * max_ops - e + 1):  # number of operators
                if e == 0:
                    s.append(0)
                elif e == 1:
                    s.append(catalans[n])
                else:
                    s.append(D[e - 1][n + 1] - D[e - 2][n + 1])
            D.append(s)
        return D

    def generate_ubi_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(0, n) = 0
            D(e, 0) = L ** e
            D(e, n) = L * D(e - 1, n) + p_1 * D(e, n - 1) + p_2 * D(e + 1, n - 1)
        """
        # enumerate possible trees
        # first generate the transposed version of D, then transpose it
        D = []
        D.append([0] + ([self.nl ** i for i in range(1, 2 * max_ops + 1)]))
        for n in range(1, 2 * max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                s.append(self.nl * s[e - 1] + self.p1 * D[n - 1][e] + self.p2 * D[n - 1][e + 1])
            D.append(s)
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        D = [[D[j][i] for j in range(len(D)) if i < len(D[j])] for i in range(max(len(x) for x in D))]
        return D

    def write_int(self, val):
        """
        Convert a decimal integer to a representation in the given base.
        The base can be negative.
        In balanced bases (positive), digits range from -(base-1)//2 to (base-1)//2
        Use a numeral decomposition scheme when relevant
        """
        base = self.int_base
        balanced = self.balanced
        res = []
        max_digit = abs(base)
        if balanced:
            max_digit = (base - 1) // 2
        else:
            if base > 0:
                neg = val < 0
                val = -val if neg else val
        while True:
            rem = val % base
            val = val // base
            if rem < 0 or rem > max_digit:
                rem -= base
                val += 1
            res.append(str(rem))
            if val == 0:
                break
        if base < 0 or balanced:
            res.append('INT')
        else:
            res.append('INT-' if neg else 'INT+')

        # If we use the decomposition scheme we must split the integer into the base 10 decomposition
        if self.numeral_decomp:
            if len(res) == 2:
                return res[::-1]
            else:
                return self.numeral_decomp_builder(res[0], res[-1], res[1:-1], 0)
        else:
            return res[::-1]

    def numeral_decomp_builder(self, int_in, int_sign, rem_int, order):
        """
        Splits a given integer (along with its overall sign) into a base 10 decomposition
        :param int_in: Input digit
        :param int_sign: Overall sign of the expression
        :param rem_int: Remaining digits to be considered
        :param order: Order in base 10 of int_in
        :return:
        """
        if int(int_in) == 0:
            return self.numeral_decomp_builder(rem_int[0], int_sign, rem_int[1:], order + 1)
        if len(rem_int) == 0:
            return_val = []
        else:
            return_val = self.numeral_decomp_builder(rem_int[0], int_sign, rem_int[1:], order + 1)
        if order == 0:
            return ['add', int_sign, int_in] + return_val
        add_term = ['add'] if len(rem_int) > 0 else []
        coeff_term = ['mul', int_sign, int_in] if (int(int_in) != 1 or int_sign != 'INT+') else []
        if order == 1:
            return add_term + coeff_term + ['INT+', '10'] + return_val
        else:
            return add_term + coeff_term + ['pow', 'INT+', '10'] + self.write_int(order) + return_val

    def parse_int(self, lst):
        """
        Parse a list that starts with an integer.
        Return the integer value, and the position it ends in the list.
        """
        base = self.int_base
        balanced = self.balanced
        val = 0
        if not (balanced and lst[0] == 'INT' or base >= 2 and lst[0] in ['INT+', 'INT-'] or base <= -2 and lst[0] == 'INT'):
            print(lst)
            raise InvalidPrefixExpression(f"Invalid integer in prefix expression")
        i = 0
        for x in lst[1:]:
            if not (x.isdigit() or x[0] == '-' and x[1:].isdigit()):
                break
            val = val * base + int(x)
            i += 1
        if base > 0 and lst[0] == 'INT-':
            val = -val
        return val, i + 1

    def sample_next_pos_ubi(self, nb_empty, nb_ops, rng):
        """
        Sample the position of the next node (unary-binary case).
        Sample a position in {0, ..., `nb_empty` - 1}, along with an arity.
        """
        assert nb_empty > 0
        assert nb_ops > 0
        probs = []
        for i in range(nb_empty):
            probs.append((self.nl ** i) * self.p1 * self.ubi_dist[nb_empty - i][nb_ops - 1])
        for i in range(nb_empty):
            probs.append((self.nl ** i) * self.p2 * self.ubi_dist[nb_empty - i + 1][nb_ops - 1])
        probs = [p / self.ubi_dist[nb_empty][nb_ops] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = rng.choice(2 * nb_empty, p=probs)
        arity = 1 if e < nb_empty else 2
        e = e % nb_empty
        return e, arity

    def get_leaf(self, max_int, rng):
        """
        Generate a leaf.
        """

        leaf_type = rng.choice(4, p=self.leaf_probs)
        if leaf_type == 0:
            return [list(self.variables.keys())[rng.randint(self.n_variables)]]

        # We do not require any coefficients
        elif leaf_type == 1:
            raise NotImplementedError

        elif leaf_type == 2:
            c = rng.randint(1, max_int + 1)
            c = c if (self.positive or rng.randint(2) == 0) else -c
            return self.write_int(c)
        else:
            return [self.constants[rng.randint(len(self.constants))]]

    def _generate_expr(self, nb_total_ops, max_int, rng, require_x=False, require_y=False, require_z=False):
        """
        Create a tree with exactly `nb_total_ops` operators.
        """
        stack = [None]
        nb_empty = 1  # number of empty nodes
        l_leaves = 0  # left leaves - None states reserved for leaves
        t_leaves = 1  # total number of leaves (just used for sanity check)

        # create tree
        for nb_ops in range(nb_total_ops, 0, -1):

            # next operator, arity and position
            skipped, arity = self.sample_next_pos_ubi(nb_empty, nb_ops, rng)
            if arity == 1:
                op = rng.choice(self.una_ops, p=self.una_ops_probs)
            else:
                op = rng.choice(self.bin_ops, p=self.bin_ops_probs)

            nb_empty += self.OPERATORS[op] - 1 - skipped  # created empty nodes - skipped future leaves
            t_leaves += self.OPERATORS[op] - 1            # update number of total leaves
            l_leaves += skipped                           # update number of left leaves

            # update tree
            pos = [i for i, v in enumerate(stack) if v is None][l_leaves]
            stack = stack[:pos] + [op] + [None for _ in range(self.OPERATORS[op])] + stack[pos + 1:]

        # sanity check
        assert len([1 for v in stack if v in self.all_ops]) == nb_total_ops
        assert len([1 for v in stack if v is None]) == t_leaves

        # create leaves
        # optionally add variables x, y, z if possible
        assert not require_z or require_y
        assert not require_y or require_x
        leaves = [self.get_leaf(max_int, rng) for _ in range(t_leaves)]
        if require_z and t_leaves >= 2:
            leaves[1] = ['z']
        if require_y:
            leaves[0] = ['y']
        if require_x and not any(len(leaf) == 1 and leaf[0] == 'x' for leaf in leaves):
            leaves[-1] = ['x']
        rng.shuffle(leaves)

        # insert leaves into tree
        for pos in range(len(stack) - 1, -1, -1):
            if stack[pos] is None:
                stack = stack[:pos] + leaves.pop() + stack[pos + 1:]
        assert len(leaves) == 0

        return stack

    def write_infix(self, token, args):
        """
        Infix representation.
        Convert prefix expressions to a format that SymPy can parse.
        """
        if token == 'add':
            return f'({args[0]})+({args[1]})'
        elif token == 'sub':
            return f'({args[0]})-({args[1]})'
        elif token == 'mul':
            return f'({args[0]})*({args[1]})'
        elif token == 'div':
            return f'({args[0]})/({args[1]})'
        elif token == 'pow':
            return f'({args[0]})**({args[1]})'
        elif token == 'polylog':
            return f'polylog({args[0]}, {args[1]})'
        # For the symbol we have to look at the number of allowed arguments
        elif token == 'citi':
            str_ret = 'citi('
            final_str = ')'
            for arg_citi in args:
                str_ret += f'{arg_citi},'
            return str_ret[:-1] + final_str
        elif token == 'rac':
            return f'({args[0]})**(1/({args[1]}))'
        elif token == 'abs':
            return f'Abs({args[0]})'
        elif token == 'inv':
            return f'1/({args[0]})'
        elif token == 'pow2':
            return f'({args[0]})**2'
        elif token == 'pow3':
            return f'({args[0]})**3'
        elif token == 'pow4':
            return f'({args[0]})**4'
        elif token == 'pow5':
            return f'({args[0]})**5'
        elif token in ['sign', 'sqrt', 'exp', 'ln']:
            return f'{token}({args[0]})'
        elif token == 'f':
            return f'f({args[0]})'
        elif token.startswith('INT'):
            return f'{token[-1]}{args[0]}'
        else:
            return token

    def _prefix_to_infix(self, expr):
        """
        Parse an expression in prefix mode, and output it in either:
          - infix mode (returns human readable string)
          - develop mode (returns a dictionary with the simplified expression)
        """
        if len(expr) == 0:
            raise InvalidPrefixExpression("Empty prefix list.")
        t = expr[0]
        if t in self.operators:
            args = []
            l1 = expr[1:]
            for _ in range(self.OPERATORS[t]):
                i1, l1 = self._prefix_to_infix(l1)
                args.append(i1)
            return self.write_infix(t, args), l1
        elif t in self.variables or t in self.constants or t == 'I':
            return t, expr[1:]
        else:
            val, i = self.parse_int(expr)
            return str(val), expr[i:]

    def prefix_to_infix(self, expr):
        """
        Prefix to infix conversion.
        """
        p, r = self._prefix_to_infix(expr)
        if len(r) > 0:
            raise InvalidPrefixExpression(f"Incorrect prefix expression \"{expr}\". \"{r}\" was not parsed.")
        return f'({p})'

    def rewrite_sympy_expr(self, expr):
        """
        Rewrite a SymPy expression.
        """
        expr_rw = expr
        for f in self.rewrite_functions:
            if f == 'expand':
                expr_rw = sp.expand(expr_rw)
            elif f == 'factor':
                expr_rw = sp.factor(expr_rw)
            elif f == 'expand_log':
                expr_rw = sp.expand_log(expr_rw, force=True)
            elif f == 'logcombine':
                expr_rw = sp.logcombine(expr_rw, force=True)
            elif f == 'powsimp':
                expr_rw = sp.powsimp(expr_rw, force=True)
            elif f == 'simplify':
                expr_rw = simplify(expr_rw, seconds=1)
        return expr_rw

    def infix_to_sympy(self, infix, no_rewrite=False, check_stack=True):
        """
        Convert an infix expression to SymPy.
        """
        expr = parse_expr(infix, evaluate=True, local_dict=self.local_dict)
        if expr.has(sp.I) or expr.has(AccumBounds):
            logger.error('Expression {} failed. Was {} originally'.format(expr, infix))
            raise ValueErrorExpression
        if not no_rewrite:
            expr = self.rewrite_sympy_expr(expr)
        return expr

    def _sympy_to_prefix(self, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        n_args = len(expr.args)

        # derivative operator
        if op == 'derivative':
            assert n_args >= 2
            assert all(len(arg) == 2 and str(arg[0]) in self.variables and int(arg[1]) >= 1 for arg in expr.args[1:]), expr.args
            parse_list = self.sympy_to_prefix(expr.args[0])
            for var, degree in expr.args[1:]:
                parse_list = ['derivative' for _ in range(int(degree))] + parse_list + [str(var) for _ in range(int(degree))]
            return parse_list

        assert (op == 'add' or op == 'mul' or op == 'citi') and (n_args >= 2) or (op != 'add' and op != 'mul') and (1 <= n_args <=2)

        # square root
        if op == 'pow' and isinstance(expr.args[1], sp.Rational) and expr.args[1].p == 1 and expr.args[1].q == 2:
            return ['sqrt'] + self.sympy_to_prefix(expr.args[0])

        # parse children
        parse_list = []
        for i in range(n_args):
            if i == 0 or (i < n_args - 1 and op != 'citi'):
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
            return self.write_int(int(str(expr)))
        elif isinstance(expr, sp.Rational):
            return ['div'] + self.write_int(int(expr.p)) + self.write_int(int(expr.q))
        elif expr == sp.E:
            return ['E']
        elif expr == sp.pi:
            return ['pi']
        elif expr == sp.I:
            return ['I']
        # SymPy operator
        for op_type, op_name in self.SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                return self._sympy_to_prefix(op_name, expr)
        # If we have a symbol
        if isinstance(expr, sp.core.function.AppliedUndef):
            if expr.name == 'citi':
                return self._sympy_to_prefix('citi', expr)

        # unknown operator
        raise UnknownSymPyOperator(f"Unknown SymPy operator: {expr}")

    def generate_argument(self, nb_ops_term_base, rng, sympy_form=True):
        """
        Generate the argument of a polylogarithmic function.
        The expected number of base operators controls the length of the expression
        :param nb_ops_term_base:
        :param rng:
        :param sympy_form:
        :return:
        """

        x = self.variables['x']
        F_expr = ['']

        # Introduce diversity in the number of base operators
        nb_ops_term = max(nb_ops_term_base + rng.randint(-2, 3), 0)
        sympy_expr = 0

        try:
            # generate an expression and rewrite it,
            # avoid issues in 0 and convert to SymPy
            while 'x' not in F_expr or sp.expand(sympy_expr) == 0 or sp.expand(sympy_expr) == sp.zoo\
                    or x not in sp.factor(sp.expand(sympy_expr)).free_symbols:

                F_expr = self._generate_expr(nb_ops_term, self.max_int, rng)
                integer_leafs = [abs(int(leaf)) for leaf in F_expr if leaf.isdigit()]

                # If the integers are bigger than the max integer allowed
                if len(integer_leafs) > 0:
                    if max(integer_leafs) > self.max_int_tot:
                        return None

                infix_expr = self.prefix_to_infix(F_expr)
                sympy_expr = self.infix_to_sympy(infix_expr)

                # Check the length and the maximum integers encountered
                set_numbers = sp.expand(sympy_expr).atoms(sp.Number)
                if len(set_numbers) > 0:
                    if max(set_numbers) > pow(self.max_int, self.max_int_tot/2)\
                            or min(set_numbers) < - pow(self.max_int, self.max_int_tot/2):
                        return None

            # Do some simple simplification of the arguments
            if sympy_form:
                return sp.factor(sp.expand(sympy_expr))
            else:
                return F_expr

        except TimeoutError:
            raise
        except (ValueErrorExpression, UnknownSymPyOperator, OverflowError, TypeError):
            return None
        except Exception as e:
            logger.error(
                "An unknown exception of type {0} occurred in line {1} for expression \"{2}\". Arguments:{3!r}.".format(
                    type(e).__name__, sys.exc_info()[-1].tb_lineno, F_expr, e.args))
            return None

    @timeout(80)
    def gen_symbol_func(self, rng, proba_arg_repeat):
        """
        Generate a polylogarithmic function whose symbol is to be computed
        :param rng:
        :param proba_arg_repeat:
        :return:
        """

        x = self.variables['x']
        # Number of terms in the arguments along with the number of distinct terms to be generated
        nb_ops_base = rng.randint(0, self.max_ops + 1)
        nb_terms = rng.randint(1, self.max_terms_transc + 1)

        F_tot = 0

        for term in range(nb_terms):

            # Factor in front
            factor = rng.randint(1, self.max_int + 1)

            # Half the time we get a negative contribution
            factor = factor if rng.randint(2) == 0 else -factor

            # We allow for fractions as arguments
            factor_type = rng.choice(3, p=self.frac_probs)

            # Inverse fraction only
            if factor_type == 1:
                factor = sp.Rational(1, factor)

            # Complete fraction
            if factor_type == 2:
                second_factor = rng.randint(1, self.max_int + 1)
                factor = sp.Rational(factor, second_factor)

            # Choose a partition for the functional skeleton given the transcendental weight
            term_type = rng.choice(len(self.transc_probs), p=self.transc_probs)
            partition_polylog = self.partitions[term_type]
            f_part = 1

            # For each term in the partition construct the appropriate polylog expression
            for polylog_weight, repet_num in partition_polylog.items():
                last_arg = None

                for j in range(repet_num):

                    # Half the time we get a negative argument
                    arg_sign = 1 if rng.randint(2) == 0 else -1

                    # If we repeat the last argument (mimics potential physical amplitudes)
                    if last_arg is not None and rng.random() < proba_arg_repeat:
                        add_arg = last_arg

                    # If we need to generate a new argument
                    else:
                        new_arg = self.generate_argument(nb_ops_base, rng)
                        if new_arg is None:
                            return None
                        else:
                            last_arg = new_arg
                        add_arg = arg_sign * new_arg

                    # Need to have a special preprocessing for logarithm to get rid of overall constants
                    # and to have them in a canonical form
                    if polylog_weight == 1:
                        log_expr_add = remove_overall_const_log(sp.log(add_arg), x, full_expand=True)
                        f_part *= sp.expand_log(log_expr_add, force=True)
                    else:
                        f_part *= sp.polylog(polylog_weight, add_arg)

            F_tot += factor * f_part

        if F_tot is None:
            return None

        F_expr = self.sympy_to_prefix(F_tot)

        # skip too long sequences
        if len(F_expr) + 2 > self.max_len:
            return None

        return ['x'], F_expr

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument("--operators", type=str, default="add:2,sub:1",
                            help="Operators (add, sub, mul, div), followed by weight")
        parser.add_argument("--max_ops", type=int, default=10,
                            help="Maximum number of operators")
        parser.add_argument("--max_terms_transc", type=int, default=10,
                            help="Maximum number transcendental terms in expression")
        parser.add_argument("--max_ops_G", type=int, default=4,
                            help="Maximum number of operators for G in IPP")
        parser.add_argument("--max_int", type=int, default=10000,
                            help="Maximum integer value")
        parser.add_argument("--int_base", type=int, default=10,
                            help="Integer representation base")
        parser.add_argument("--balanced", type=bool_flag, default=False,
                            help="Balanced representation (base > 0)")
        parser.add_argument("--precision", type=int, default=10,
                            help="Float numbers precision")
        parser.add_argument("--positive", type=bool_flag, default=False,
                            help="Do not sample negative numbers")
        parser.add_argument("--rewrite_functions", type=str, default="",
                            help="Rewrite expressions with SymPy")
        parser.add_argument("--leaf_probs", type=str, default="0.75,0,0.25,0",
                            help="Leaf probabilities of being a variable, a coefficient, an integer, or a constant.")
        parser.add_argument("--n_variables", type=int, default=1,
                            help="Number of variables in expressions (between 1 and 4)")
        parser.add_argument("--n_coefficients", type=int, default=0,
                            help="Number of coefficients in expressions (between 0 and 10)")

    def create_train_iterator(self, task, params, data_path):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=True,
            rng=None,
            params=params,
            path=(None if data_path is None else data_path[task][0])
        )
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 1800),
            batch_size=params.batch_size,
            num_workers=(params.num_workers if data_path is None or params.num_workers == 0 else 1),
            shuffle=False,
            collate_fn=dataset.collate_fn
        )

    def create_test_iterator(self, data_type, task, params, data_path):
        """
        Create a dataset for this environment.
        """
        assert data_type in ['valid', 'test']
        logger.info(f"Creating {data_type} iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=False,
            rng=np.random.RandomState(0),
            params=params,
            path=(None if data_path is None else data_path[task][1 if data_type == 'valid' else 2])
        )
        return DataLoader(
            dataset,
            timeout=0,
            batch_size=params.batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )


class EnvDataset(Dataset):

    def __init__(self, env, task, train, rng, params, path):
        super(EnvDataset).__init__()
        # self.env = env
        self.params = params
        self.rng = rng
        self.train = train
        self.task = task
        self.batch_size = params.batch_size
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.global_rank = params.global_rank
        self.count = 0
        self.n_gpu_per_node = params.n_gpu_per_node
        self.local_rank = params.local_rank
        self.reload_size = params.reload_size
        assert (train is True) == (rng is None)
        assert task in CharSPEnvironment.TRAINING_TASKS

        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size
        self.same_nb_ops_per_batch = params.same_nb_ops_per_batch

        # dataset size: infinite iterator for train, finite for valid / test (default of 5000 if no file provided)
        if self.train:
            self.size = 1 << 60
        else:
            assert os.path.isfile(self.path)
            logger.info(f"Preparing to load data from {self.path} ...")
            with io.open(self.path, mode='r', encoding='utf-8') as f:
                lines = [line.rstrip().split('|') for line in f]
            self.size = 5000 if path is None else len(lines)

    def open_dataset(self):

        # Define the environment here for multiprocessing issues
        self.env = CharSPEnvironment(self.params)
        # generation, or reloading from file
        if self.path is not None:
            assert os.path.isfile(self.path)
            logger.info(f"Loading data from {self.path} ...")
            with io.open(self.path, mode='r', encoding='utf-8') as f:
                # either reload the entire file, or the first N lines (for the training set)
                if not self.train:
                    lines = [line.rstrip().split('|') for line in f]
                else:
                    lines = []
                    for i, line in enumerate(f):
                        if i == self.reload_size:
                            break
                        if i % self.n_gpu_per_node == self.local_rank:
                            lines.append(line.rstrip().split('|'))
            self.data = [xy.split('\t') for _, xy in lines]
            self.data = [xy for xy in self.data if len(xy) == 2]
            logger.info(f"Loaded {len(self.data)} equations from the disk.")

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        x, y = zip(*elements)
        nb_ops = [sum(int(word in self.env.OPERATORS) for word in seq) for seq in x]
        x = [torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id]) for seq in x]
        y = [torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id]) for seq in y]
        x, x_len = self.env.batch_sequences(x)
        y, y_len = self.env.batch_sequences(y)
        return (x, x_len), (y, y_len), torch.LongTensor(nb_ops)

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if self.rng is None:
            assert self.train is True
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            self.rng = np.random.RandomState([worker_id, self.global_rank, self.env_base_seed])
            logger.info(f"Initialized random generator for worker {worker_id}, with seed {[worker_id, self.global_rank, self.env_base_seed]} (base seed={self.env_base_seed}).")

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0)
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        if not hasattr(self, 'env'):
            self.open_dataset()
        self.init_rng()
        if self.path is None:
            return self.generate_sample()
        else:
            return self.read_sample(index)

    def read_sample(self, index):
        """
        Read a sample.
        """
        if self.train:
            index = self.rng.randint(len(self.data))
        x, y = self.data[index]
        x = x.split()
        y = y.split()
        assert len(x) >= 1 and len(y) >= 1
        return x, y

    def generate_sample(self):
        """
        Generate a sample for the symbol integration task
        For the functional simplification task data generation can be done using the RL data generator
        """

        while True:

            try:
                if self.task == 'symbol_int':
                    xy = self.env.gen_symbol_func(self.rng, self.params.proba_arg_repeat)
                else:
                    raise Exception(f'Unknown data type: {self.task}')
                if xy is None:
                    continue
                x, y = xy
                break
            except TimeoutError:
                continue
            except Exception as e:
                logger.error("An unknown exception of type {0} occurred for worker {4} in line {1} for expression \"{2}\". Arguments:{3!r}.".format(type(e).__name__, sys.exc_info()[-1].tb_lineno, 'F', e.args, self.get_worker_id()))
                continue

        self.count += 1

        # clear SymPy cache periodically
        if CLEAR_SYMPY_CACHE_FREQ > 0 and self.count % CLEAR_SYMPY_CACHE_FREQ == 0:
            logger.warning(f"Clearing SymPy cache (worker {self.get_worker_id()})")
            clear_cache()

        return x, y
