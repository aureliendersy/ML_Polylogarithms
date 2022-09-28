"""
Routines helping the definition of the RL environment and the data generation
"""

import numpy as np
import sympy as sp
from sympy import Poly
import random
import matplotlib.pyplot as plt
import time
from itertools import combinations
from observations.utils_obs import ACTION_COEFF_MAP, INC_ACTIONS
from model.classical_algorithm import bfs


def cyclic_perm_arg(expr):
    """For a sympy expression that is made up by a linear combination
    of terms, we do a cyclic permutation between these arguments"""

    if not isinstance(expr, sp.Add):
        raise TypeError('Can only shift an additive expression')
    else:
        new_args = expr.args[1:] + expr.args[0:1]
        new_expr = expr._new_rawargs(*new_args)
        return new_expr


def get_first_polylog_obs(expr):
    """For a sympy expression return the first polylog term
    Overly cautious here but shouldn't hurt"""

    # To retain the polylog term we have to verify the type of expression we are being fed
    if isinstance(expr, sp.Add):
        first_arg = expr.args[0]
        if isinstance(first_arg, sp.Mul):
            polylog_term = first_arg.args[-1]
        elif isinstance(first_arg, sp.polylog):
            polylog_term = first_arg
        else:
            raise TypeError('Error when finding first polylog term of sequence {}.'
                            'The term {} is not recognized properly'.format(str(expr), str(first_arg)))
    elif isinstance(expr, sp.Mul):
        second_arg = expr.args[-1]
        if isinstance(second_arg, sp.polylog):
            polylog_term = second_arg
        else:
            raise TypeError('Error when finding first polylog term of sequence {}.'
                            'The term {} is not recognized properly'.format(str(expr), str(second_arg)))
    elif isinstance(expr, sp.polylog):
        polylog_term = expr
    else:
        raise ValueError('Error when finding first polylog term of sequence {}.'.format(str(expr)))
    return polylog_term


def get_random_polylog_obs(expr):
    """For a sympy expression return a random polylog term"""

    # Need to identify the polylogs first
    if isinstance(expr, sp.Add):
        random_arg = random.choice(expr.args)
        if isinstance(random_arg, sp.Mul):
            polylog_term = random_arg.args[-1]
        elif isinstance(random_arg, sp.polylog):
            polylog_term = random_arg
        else:
            raise TypeError('Error when finding first polylog term of sequence {}.'
                            'The term {} is not recognized properly'.format(str(expr), str(random_arg)))
    elif isinstance(expr, sp.Mul):
        second_arg = expr.args[-1]
        if isinstance(second_arg, sp.polylog):
            polylog_term = second_arg
        else:
            raise TypeError('Error when finding first polylog term of sequence {}.'
                            'The term {} is not recognized properly'.format(str(expr), str(second_arg)))
    elif isinstance(expr, sp.polylog):
        polylog_term = expr
    else:
        raise ValueError('Error when finding first polylog term of sequence {}.'.format(str(expr)))
    return polylog_term


def get_inv_arg_poly(polylog_expr, simple_form='cancel'):
    """Return the polylog expression where the argument is inverted"""

    if not isinstance(polylog_expr, sp.polylog):
        raise TypeError('Expected {} to be a polylog expression'.format(polylog_expr))

    # Take the inverse of the argument and simplify with either cancel or factor
    polylog_arg = polylog_expr.args[-1]
    if simple_form == 'cancel':
        return_expr = polylog_expr.replace(polylog_arg, sp.cancel(1 / polylog_arg))
    else:
        return_expr = polylog_expr.replace(polylog_arg, sp.factor(1/polylog_arg))
    return return_expr


def get_refl_arg_poly(polylog_expr, simple_form='cancel'):
    """Return the polylog expression where the argument is reflected"""

    if not isinstance(polylog_expr, sp.polylog):
        raise TypeError('Expected {} to be a polylog expression'.format(polylog_expr))

    # Take 1-the argument and simplify with either cancel or factor
    polylog_arg = polylog_expr.args[-1]
    if simple_form == 'cancel':
        return_expr = polylog_expr.replace(polylog_arg, sp.cancel(1 - polylog_arg))
    else:
        return_expr = polylog_expr.replace(polylog_arg, sp.factor(1-polylog_arg))
    return return_expr


def get_dupli_arg_poly(polylog_expr, simple_form='cancel'):
    """Act with the duplication action for the dilogarithms"""

    if not isinstance(polylog_expr, sp.polylog):
        raise TypeError('Expected {} to be a polylog expression'.format(polylog_expr))

    # Need to add two terms now due to the form of the identity
    polylog_arg = polylog_expr.args[-1]
    try:
        if simple_form == 'cancel':
            new_expr = sp.Rational(1, 2) * polylog_expr.replace(polylog_arg, sp.cancel(polylog_arg * polylog_arg))
            new_expr = new_expr - polylog_expr.replace(polylog_arg, sp.cancel(-polylog_arg))
        else:
            new_expr = sp.Rational(1, 2)*polylog_expr.replace(polylog_arg, sp.factor(polylog_arg*polylog_arg))
            new_expr = new_expr - polylog_expr.replace(polylog_arg, sp.factor(-polylog_arg))
    except:
        print('{} was the argument to square'.format(polylog_arg))
        print('{} was the original expression'.format(polylog_expr))
        return None

    return new_expr


def act_arg(arg, action_name, simple_form='cancel'):
    """
    Return the argument which might lead to a cyclic trajectory if
    acted twice upon
    """

    # Need to feed an explicit action name
    if action_name == 'inversion':
        if simple_form == 'cancel':
            return sp.cancel(1 / arg)
        else:
            return sp.factor(1 / arg)
    elif action_name == 'reflection':
        if simple_form == 'cancel':
            return sp.cancel(1 - arg)
        else:
            return sp.factor(1 - arg)
    elif action_name == 'duplication':
        if simple_form == 'cancel':
            return sp.cancel(- arg)
        else:
            return sp.factor(- arg)
    elif action_name == 'cyclic':
        return arg
    else:
        raise NameError('Acting on argument with {} is not implemented'.format(action_name))


def act_arg_poly(polylog_exr, action_name, simple_form='cancel'):
    """
    Act on the argument of a polylog
    :param polylog_exr:
    :param action_name:
    :param simple_form: Simplification form, sp.cancel or sp.factor
    :return:
    """

    # Need to feed an explicit action name
    if action_name == 'inversion':
        return get_inv_arg_poly(polylog_exr, simple_form)
    elif action_name == 'reflection':
        return get_refl_arg_poly(polylog_exr, simple_form)
    elif action_name == 'duplication':
        return get_dupli_arg_poly(polylog_exr, simple_form)
    elif action_name == 'cyclic':
        return polylog_exr
    else:
        raise NameError('Acting on argument with {} is not implemented'.format(action_name))


def act_poly_order(polylog_expr, action_list, action_num):
    """
    Given an action list, pick an action at random at start by acting with this one
    then act with the others in order
    :param polylog_expr:
    :param action_list:
    :param action_num:
    :return:
    """

    act_start = random.randint(0, len(action_list)-1)
    current_act = act_start
    acted = 0
    expr_ret = polylog_expr
    coeff_mul = 1

    while acted < action_num:
        expr_ret = act_arg_poly(expr_ret, action_list[current_act])
        acted += 1
        current_act = (current_act+1) % len(action_list)

        # Be careful about adding the correct overall coefficient
        coeff_mul *= ACTION_COEFF_MAP[action_list[current_act]]

    return expr_ret, coeff_mul


def act_poly_random(polylog_expr, action_list, action_num, simple_form='cancel', verbose=False):
    """
    Pick actions from the action list and act at random on the polylog expression
    :param polylog_expr:
    :param action_list:
    :param action_num:
    :param simple_form:
    :param verbose:
    :return:
    """

    if action_num == 0:
        return polylog_expr, 1

    acted = 0
    expr_ret = polylog_expr

    # Act at random the relevant number of times
    while acted < action_num:
        action_id = random.randint(0, len(action_list) - 1)
        if verbose:
            print('Using Action {}'.format(action_list[action_id]))
        old_polylog_term = get_random_polylog_obs(expr_ret)
        new_polylog_term = act_arg_poly(old_polylog_term, action_list[action_id], simple_form)

        # Be careful about adding the correct overall coefficient
        expr_ret = expr_ret.replace(old_polylog_term, new_polylog_term*ACTION_COEFF_MAP[action_list[action_id]])
        acted += 1

    return expr_ret, 1


def act_poly_random_v2(polylog_expr, action_list, action_num, simple_form='cancel', verbose=True):
    """
    Pick actions from the action list and act at random on the polylog expression
    Add checks on the validity of the action
    :param polylog_expr:
    :param action_list:
    :param action_num:
    :param simple_form:
    :param verbose:
    :return:
    """

    if action_num == 0:
        return polylog_expr, 1

    acted = 0
    past_action_id = -1
    old_arg = 0
    expr_ret = polylog_expr

    while acted < action_num:
        old_polylog_term = get_random_polylog_obs(expr_ret)
        new_arg = old_polylog_term.args[-1]

        # Ask of the action is valid : whether we are just repeating the last action used on the same argument
        # Avoid this situation so that the current scrambling move doesn't undo the previous scrambling move
        valid_action = False
        while not valid_action:
            action_id = random.randint(0, len(action_list) - 1)
            valid_action = not(action_id == past_action_id and new_arg == old_arg)
        if verbose:
            print('Using Action {}'.format(action_list[action_id]))
        new_polylog_term = act_arg_poly(old_polylog_term, action_list[action_id], simple_form)
        expr_ret = expr_ret.replace(old_polylog_term, new_polylog_term*ACTION_COEFF_MAP[action_list[action_id]])
        acted += 1
        past_action_id = action_id
        old_arg = act_arg(new_arg, action_list[action_id], simple_form)

    return expr_ret, 1


def generate_random_argument(max_degree, max_coeff, variable, simple_form='cancel'):
    """
    Generate a random coefficient that takes the form of a rational fraction
    :param max_degree:
    :param max_coeff:
    :param variable:
    :param simple_form:
    :return:
    """

    # Generate the degree of the polynomials
    deg_num = random.randint(0, max_degree)
    deg_denom = random.randint(0, max_degree)

    # Generate the coefficients
    coeffs_num = [random.randint(-max_coeff, max_coeff) for _ in range(deg_num+1)]
    coeffs_denom = [random.randint(-max_coeff, max_coeff) for _ in range(deg_denom + 1)]

    if all(coeff == 0 for coeff in coeffs_num):
        coeffs_num = [1]

    if all(coeff == 0 for coeff in coeffs_denom):
        coeffs_denom = [1]

    # Have some consistent way of simplifying the arguments
    if simple_form == 'cancel':
        fraction_ret = sp.cancel(Poly(coeffs_num, variable) / Poly(coeffs_denom, variable))
    else:
        fraction_ret = sp.factor(Poly(coeffs_num, variable)/Poly(coeffs_denom, variable))

    # If the argument is a constant we try again
    if variable in fraction_ret.free_symbols:
        return fraction_ret
    else:
        return generate_random_argument(max_degree, max_coeff, variable, simple_form=simple_form)


def generate_null_start_state(add_terms, max_scr, max_degree, max_coeff, variable, action_list):
    """
    Generator of equations that can all explicitly be reduced down to 0
    The idea is to add 0 with the use of the identities and then to scramble the individual terms with further uses of
    said identities.
    :param add_terms:
    :param max_scr:
    :param max_degree:
    :param max_coeff:
    :param variable:
    :param action_list:
    :return:
    """

    expr_generated = 0

    for _ in range(add_terms):

        # Generate the dilog argument and overall constant
        random_arg = generate_random_argument(max_degree, max_coeff, variable, simple_form='cancel')
        random_coeff = random.randint(1, max_coeff * 4)

        # We will add 0 as term1 - term2 =0 , where both terms get scrambled
        # Pick the number of scrambles to apply on each term
        term1, term2 = sp.polylog(2, random_arg), sp.polylog(2, random_arg)
        scr1, scr2 = random.randint(1, max_scr),  random.randint(1, max_scr)

        term1, _ = act_poly_random(term1, action_list, scr1, simple_form='cancel')
        term2, _ = act_poly_random(term2, action_list, scr2, simple_form='cancel')

        expr_generated += term1*random_coeff - term2*random_coeff

    return expr_generated


def partition(number, size, min_one=True):
    """Helper function for partitioning integers"""
    n = number + size - 1
    partition_return = []
    for splits in combinations(range(n), size - 1):
        part = [s1 - s0 - 1 for s0, s1 in zip((-1,) + splits, splits + (n,))]
        if 0 not in part or not min_one:
            partition_return.append(part)
    return partition_return


def scr_null_state(num_scr, num_add_zero, max_degree, max_coeff, variable, action_list, verbose=False):
    """
    Generate a null state and scramble it with a total number of scrambles
    :param num_scr:
    :param num_add_zero:
    :param max_degree:
    :param max_coeff:
    :param variable:
    :param action_list:
    :param verbose:
    :return:
    """
    expr_generated = 0
    if num_scr < num_add_zero:
        num_add_zero = num_scr

    # Choose how to distribute the number of scrambles amongst the added terms
    part = random.choice(partition(num_scr, num_add_zero))

    for i in range(num_add_zero):
        # We will add 0 as term1 - term2 =0 , where both terms get scrambled
        random_arg = generate_random_argument(max_degree, max_coeff, variable, simple_form='cancel')
        random_coeff = random.randint(1, max_coeff * 4)
        term1, term2 = sp.polylog(2, random_arg), sp.polylog(2, random_arg)

        # Choose how to distribute the number of scrambles amongst the two terms
        scrambles = random.choice(partition(part[i], 2, min_one=False))

        if verbose:
            print('Start from {}'.format(term1*random_coeff))
            print('Scramble left side {} times and right side {} times'.format(scrambles[0], scrambles[1]))

        term1, _ = act_poly_random(term1, action_list, scrambles[0], simple_form='cancel', verbose=verbose)
        term2, _ = act_poly_random(term2, action_list, scrambles[1], simple_form='cancel', verbose=verbose)

        expr_generated += term1 * random_coeff - term2 * random_coeff

    return expr_generated


def scr_null_state_v2(num_scr, num_add_zero, max_degree, max_coeff, variable, action_list, verbose=False):
    """
    Generate a null state and scramble it with a total number of scrambles.
    Scramble only one of the added terms.
    :param num_scr:
    :param num_add_zero:
    :param max_degree:
    :param max_coeff:
    :param variable:
    :param action_list:
    :param verbose:
    :return:
    """
    expr_generated = 0
    if num_scr < num_add_zero:
        num_add_zero = num_scr

    # Choose how to distribute the number of scrambles amongst the added terms
    part = random.choice(partition(num_scr, num_add_zero))

    for i in range(num_add_zero):
        # We will add 0 as term1 - term2 =0 , where one term get scrambled
        random_arg = generate_random_argument(max_degree, max_coeff, variable, simple_form='cancel')
        random_coeff = random.randint(1, max_coeff * 4)
        term1, term2 = sp.polylog(2, random_arg), sp.polylog(2, random_arg)
        if verbose:
            print('Start from {}'.format(term1*random_coeff))
        if verbose:
            print('Scramble left side {} times and right side {} times'.format(part[i], 0))

        # Only scramble the first term
        term1, _ = act_poly_random_v2(term1, action_list, part[i], simple_form='cancel', verbose=verbose)
        term2, _ = act_poly_random_v2(term2, action_list, 0, simple_form='cancel', verbose=verbose)

        expr_generated += term1 * random_coeff - term2 * random_coeff

    return expr_generated


def generate_random_starting_eq(max_length_simple, max_length_scr, max_scr, max_degree, max_coeff, variable, proba_new,
                                action_list):
    """
    Generate a linear combination of dilogarithms, which simplifies down to an expression with max_length_simple terms

    :param max_length_simple: Maximum number of terms in the simple expression
    :param max_length_scr: Maximum number of terms in the scrambled expressions
    :param max_scr: maximum number of times we can scrambled each argument
    :param max_degree: Maximum degree that an argument can have
    :param max_coeff: Maximum coefficient that we can have in an argument. Also use for maximum overall coeff
    :param variable: Variable to use, typically x
    :param proba_new: Probability that a term in the scrambled expression has a new argument and
    has to cancel out with another term in some way
    :param action_list: actions that we can use to scramble the initial expression
    :return:
    """

    # Typically for RL we will always consider max_length_simple=0
    num_terms_simple = random.randint(0, max_length_simple)

    terms_left_scr = random.randint(max(num_terms_simple, 1), max_length_scr)

    # This is the loop that we will typically use for a standard generation
    # Define the maximum number of scrambles and add zero by randomling shuffling each turn
    if num_terms_simple == 0:
        expr_ret = generate_null_start_state(terms_left_scr, max_scr, max_degree, max_coeff, variable, action_list)

        # If simplifies to 0 explicitly relaunch
        if expr_ret == 0:
            return generate_random_starting_eq(max_length_simple, max_length_scr, max_scr,
                                               max_degree, max_coeff, variable, proba_new, action_list)
        else:
            return expr_ret, 0

    # Do not usually refer to this - In case we want to generate starts different from 0 for RL
    if num_terms_simple > 0:
        simple_args = [generate_random_argument(max_degree, max_coeff, variable) for _ in range(num_terms_simple)]
    else:
        simple_args = None

    expr_ret = 0
    simple_expr = 0

    while terms_left_scr > 0:

        # If term simplifies to a term already in the simple expression
        if (random.uniform(0, 1) > proba_new or terms_left_scr == 1) and simple_args is not None:

            random_coeff = random.randint(1, max_coeff*4)
            random_coeff = random_coeff if random.uniform(0, 1) > 0.5 else - random_coeff

            arg_simple = random.choice(simple_args)
            polylog_exp_tmp = sp.polylog(2, arg_simple)
            scr_times = random.randint(1, max_scr)

            if any(action in INC_ACTIONS for action in action_list):
                polylog_exp_tmp, coeff_mul = act_poly_random(polylog_exp_tmp, action_list, scr_times)
            else:
                polylog_exp_tmp, coeff_mul = act_poly_order(polylog_exp_tmp, action_list, scr_times)

            expr_ret += random_coeff * coeff_mul * polylog_exp_tmp
            simple_expr += random_coeff * sp.polylog(2, arg_simple)
            terms_left_scr -= 1

        # Term get created with a different argument (must generate two terms with opposite coeffs)
        else:
            random_arg = generate_random_argument(max_degree, max_coeff, variable)
            random_coeff = random.randint(1, max_coeff * 4)

            polylog_exp_tmp = sp.polylog(2, random_arg)
            polylog_exp_tmp1, polylog_exp_tmp2 = polylog_exp_tmp, polylog_exp_tmp

            scr1 = random.randint(1, max_scr)
            scr2 = random.randint(1, max_scr)
            if scr2 == scr1 and not any(action in INC_ACTIONS for action in action_list):
                scr2 += 1

            if any(action in INC_ACTIONS for action in action_list):
                polylog_exp_tmp1, coeff_mul1 = act_poly_random(polylog_exp_tmp1, action_list, scr1)
                polylog_exp_tmp2, coeff_mul2 = act_poly_random(polylog_exp_tmp2, action_list, scr2)
            else:
                polylog_exp_tmp1, coeff_mul1 = act_poly_order(polylog_exp_tmp1, action_list, scr1)
                polylog_exp_tmp2, coeff_mul2 = act_poly_order(polylog_exp_tmp2, action_list, scr2)

            expr_ret += random_coeff * coeff_mul1 * polylog_exp_tmp1 - random_coeff * coeff_mul2 * polylog_exp_tmp2
            terms_left_scr -= 2

    if expr_ret == 0:
        return generate_random_starting_eq(max_length_simple, max_length_scr, max_scr,
                                           max_degree, max_coeff, variable, proba_new, action_list)
    else:
        return expr_ret, simple_expr


def generate_transformer_starts(num_terms_simple, num_zero, max_scr, max_degree, max_coeff, variable, action_list):
    """
    Generate linear combination of dilogarithms that need not simplify down to 0.
    Will be used for the transformer data generation

    :param num_terms_simple: Number of terms in the simple expression
    :param max_scr: Total number of times we can scramble
    :param max_degree: Maximum degree that an argument can have
    :param max_coeff: Maximum coefficient that we can have in an argument. Also use for maximum overall coeff
    :param variable: Variable to use, typically x
    :param num_zero: Maximum number of time that we can add zero
    :param action_list: actions that we can use to scramble the initial expression
    :return:
    """

    expr_ret = 0
    simple_expr = 0

    # If we want to generate a term that simplifies to 0
    if num_terms_simple == 0:
        return scr_null_state_v2(max_scr, num_zero, max_degree, max_coeff, variable, action_list), 0

    # If we want to create a simple expression and then shuffle it
    simple_args = [generate_random_argument(max_degree, max_coeff, variable) for _ in range(num_terms_simple)]

    # Number of scrambles devoted to shuffle the simple term and adding zero
    zero_scr = random.randint(num_zero, max_scr) if num_zero > 0 else 0
    non_zero_scr = max_scr - zero_scr

    if zero_scr > 0:
        zero_terms = scr_null_state_v2(zero_scr, num_zero, max_degree, max_coeff, variable, action_list)
        expr_ret += zero_terms

    scr_non_zero = random.choice(partition(non_zero_scr, num_terms_simple, min_one=False))

    # Shuffle the simple term
    for j, arg in enumerate(simple_args):
        random_coeff = random.randint(1, max_coeff * 4)
        random_coeff = random_coeff if random.uniform(0, 1) > 0.5 else - random_coeff
        polylog_exp_tmp = sp.polylog(2, arg)
        polylog_exp_tmp, coeff_mul = act_poly_random_v2(polylog_exp_tmp, action_list, scr_non_zero[j], verbose=False)
        expr_ret += random_coeff * coeff_mul * polylog_exp_tmp
        simple_expr += random_coeff * sp.polylog(2, arg)

    return expr_ret, simple_expr


def load_starting_set(file_starts, logger=None):
    """
    Load the scrambled and corresponding simple equations from the starting set
    :param file_starts:
    :param logger:
    :return:
    """
    if logger is not None:
        logger.info('Loading initial eqs from {}'.format(file_starts))
    start_f = open(file_starts, "r")
    initial_eqs = start_f.readlines()
    simple_expr = [sp.sympify(eqs[:-1].split(': ')[-1]) for eqs in initial_eqs[1::4]]
    scr_expr = [sp.sympify(eqs[:-1].split(': ')[-1]) for eqs in initial_eqs[2::4]]

    return [scr_expr, simple_expr]


def load_curriculum(file_list, logger=None):
    """
    Load all of the equations from a list of files
    :param file_list:
    :param logger:
    :return:
    """
    if logger:
        logger.info('Curriculum learning')
    scr_expr = []
    simple_expr = []

    for file in file_list:
        scr_expr_file, simple_expr_file = load_starting_set(file, logger=logger)
        scr_expr.append(scr_expr_file)
        simple_expr.append(simple_expr_file)

    return [scr_expr, simple_expr]


def load_sp_eqs(start_set, num_eqs_load=None):
    """Read the starting set and parse to sympy equations"""
    print("Loading equations from {}".format(start_set))
    start_f = open(start_set, "r")
    if not num_eqs_load:
        initial_eqs = start_f.readlines()
    else:
        initial_eqs = start_f.readlines()[:num_eqs_load]
    split_eqs = [eq.split('\t') for eq in initial_eqs]
    parsed_scrambled = [sp.parse_expr(eq[0]) for eq in split_eqs]
    parsed_simplified = [sp.parse_expr(eq[1]) for eq in split_eqs]

    return parsed_scrambled, parsed_simplified


def load_prefix_eqs(start_set, num_eqs_load=None):
    """Read the starting set and parse to prefix form"""
    print("Loading equations from {}".format(start_set))
    start_f = open(start_set, "r")
    if not num_eqs_load:
        initial_eqs = start_f.readlines()
    else:
        initial_eqs = start_f.readlines()[:num_eqs_load]
    split_eqs = [eq.split('|') for eq in initial_eqs]
    info_scr = [eq[0].split(',')[-1] for eq in split_eqs]
    eqs_data = [eq[1] for eq in split_eqs]
    scr_eq = [eq.split('\t')[0] for eq in eqs_data]

    return info_scr, scr_eq


def print_full_stats_set(start_set, num_eqs_load=None):
    """Get information on the scrambling parameters used to generate the set
    (Relevant for transformer data)"""
    info_scr, scr_eqs = load_prefix_eqs(start_set, num_eqs_load=num_eqs_load)
    print("Looking at stats over {} equations".format(len(scr_eqs)))

    len_scr = np.array([eq.count('polylog') for eq in scr_eqs])
    len_simple = [int(info[info.index('zeros')-1]) for info in info_scr]

    num_scr = [int(info.split('scrambles')[-1]) for info in info_scr]
    num_zeros = [int(info[info.index('scrambles')-1]) for info in info_scr]

    distr_scr_len = dict(zip(*np.unique(len_scr, return_counts=True)))
    distr_simple = dict(zip(*np.unique(len_simple, return_counts=True)))
    distr_scr = dict(zip(*np.unique(num_scr, return_counts=True)))
    distr_zeros = dict(zip(*np.unique(num_zeros, return_counts=True)))

    plt.bar(*zip(*distr_scr_len.items()))
    plt.title('Number of polylogs in the scrambled expressions')
    plt.show()
    plt.title('Number of polylogs in the simplified expressions')
    plt.bar(*zip(*distr_simple.items()))
    plt.show()
    plt.title('Total number of scrambles used')
    plt.bar(*zip(*distr_scr.items()))
    plt.show()
    plt.title('Number of times zero is added')
    plt.bar(*zip(*distr_zeros.items()))
    plt.show()


def print_stats_set(start_set, num_eqs_load=None):
    """Get information on the length of the inputs and desired outputs
    (Relevant for transformer data)"""
    scr_eqs, simple_eqs = load_sp_eqs(start_set, num_eqs_load=num_eqs_load)
    print("Looking at stats over {} equations".format(len(scr_eqs)))

    len_scr = np.array([eq.count(sp.polylog) for eq in scr_eqs])
    len_simple = np.array([eq.count(sp.polylog) for eq in simple_eqs])

    distr_scr = dict(zip(*np.unique(len_scr, return_counts=True)))
    distr_simple = dict(zip(*np.unique(len_simple, return_counts=True)))

    plt.bar(*zip(*distr_scr.items()))
    plt.title('Number of polylogs in the scrambled expressions')
    plt.show()
    plt.title('Number of polylogs in the simplified expressions')
    plt.bar(*zip(*distr_simple.items()))
    plt.show()


if __name__ == '__main__':

    # To print the scrambling and complexity information of a dataset relevant for the transformer model
    print_full_stats_set('example_path')
    exit()
