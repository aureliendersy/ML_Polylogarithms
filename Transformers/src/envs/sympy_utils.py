# Adapted from https://github.com/facebookresearch/SymbolicMathematics under CC BY-NC 4.0

"""
Contains required sympy routines
"""


from logging import getLogger
import sympy as sp
import numpy as np
from sympy.utilities.iterables import partitions

from ..utils import timeout, TimeoutError

logger = getLogger()


def simplify(f, seconds):
    """
    Simplify an expression.
    """
    assert seconds > 0
    @timeout(seconds)
    def _simplify(f):
        try:
            f2 = sp.simplify(f)
            if any(s.is_Dummy for s in f2.free_symbols):
                logger.warning(f"Detected Dummy symbol when simplifying {f} to {f2}")
                return f
            else:
                return f2
        except TimeoutError:
            return f
        except Exception as e:
            logger.warning(f"{type(e).__name__} exception when simplifying {f}")
            return f
    return _simplify(f)


def remove_overall_const_log(expr, variable, full_expand=True):
    """
    Take an expression that contains logs and remove from them any overall constant
    :param expr:
    :param variable:
    :param full_expand:
    :return:
    """

    expr_expand = sp.expand_log(expr, force=True)
    if len(sp.Add.make_args(expr_expand)) > 1:
        x_dependant_expr = expr_expand.as_independent(variable)[-1]
    else:
        x_dependant_expr = expr_expand

    if full_expand:
        return x_dependant_expr
    else:
        return sp.logcombine(x_dependant_expr, force=True)


def simplify_log_squared(log1, log2):
    """
    Take as input two logarithmic expression and the variable they depend on
    The goal is to simplify/standardize their product so that we only generate
    a unique representative. Output in expanded form

    e.g: going from log(x)*log(x^2) to 2*log(x)^2

    :param log1:
    :param log2:
    :return:
    """

    log1_expand = sp.expand_log(log1, force=True)
    log2_expand = sp.expand_log(log2, force=True)

    log_combine = sp.expand(log1_expand*log2_expand)

    return log_combine


def convert_mma_sympy(mma_expr):
    """
    For a given MMA expression obtained as a string, we parse it into a readable sympy expr

    :param mma_expr:
    :return:
    """
    mma_expr = mma_expr.replace('[', '(').replace(']', ')').replace('^', '**').lower()

    return sp.sympify(mma_expr)


def gen_proba_transcendental(trans_weight, skew_basis_proba):
    """For a given transcendental weight give the list of partitions
    and the probability to associate to each"""

    list_p = []
    proba_p = []

    for size_p, partition in partitions(trans_weight, size=True):
        list_p.append(partition)
        proba_p.append(trans_weight+1-size_p*skew_basis_proba)

    proba_np = np.array(proba_p).astype(np.float64)
    proba_np = proba_np / proba_np.sum()

    return list_p, proba_np


def convert_mma_file_to_sympy(filepath, add_name=''):
    """

    For a given file, open it and create a new file where we convert each mma expression to a sympy one

    :param filepath:
    :param add_name:
    :return:
    """

    mma_functions = open(filepath, "r")
    mma_lines = mma_functions.readlines()

    sympy_file = open(filepath.replace('mma', 'sp' + add_name), "w+")

    for line_num, line_s in enumerate(mma_lines):
        sympy_file.write(f'{str(convert_mma_sympy(line_s))}\n')
        sympy_file.flush()

        percent_done = line_num / len(mma_lines) * 100
        if line_num % 1000 == 0:
            print("Converting MMA file to sympy --->  {}%".format(str(round(percent_done))))
            logger.info("Converting MMA file to sympy --->  {}%".format(str(round(percent_done))))

    sympy_file.close()