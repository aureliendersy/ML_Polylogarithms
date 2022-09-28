"""
Contains helper routines for the observation classes
"""


import sympy as sp

ACTION_COEFF_MAP = {'inversion': -1, 'reflection': -1, 'duplication': 1}

VAR_LIST = ['x']

SYMB_LIST = ['INT+', 'INT-']

BASE_SYMPY_OPERATORS = {
        # Elementary functions
        sp.Add: 'add',
        sp.Mul: 'mul',
        sp.Pow: 'pow',
        # Polylogs
        sp.polylog: 'polylog',
    }

BASE_OPERATORS = {
        # Elementary functions
        'add': 2,
        'mul': 2,
        'div': 2,
        'pow': 2,
        # Polylogs
        'polylog': 2,
    }

# Actions that can potentially increase the number of polylogs
INC_ACTIONS = ['duplication', 'root3', '5term']


def write_int(val, base=10, numeral_decomp=False):
    """
    Convert a decimal integer to a representation in the given base.
    """

    res = []
    max_digit = abs(base)
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

    res.append('INT-' if neg else 'INT+')
    if numeral_decomp:
        if len(res) == 2:
            return res[::-1]
        else:
            return numeral_decomp_builder(res[0], res[-1], res[1:-1], 0)
    else:
        return res[::-1]


def numeral_decomp_builder(int_in, int_sign, rem_int, order):
    """Write an integer in base 10 and decompose it
    e.g 26 = 6 + 2*10"""
    if int(int_in) == 0:
        return numeral_decomp_builder(rem_int[0], int_sign, rem_int[1:], order + 1)
    if len(rem_int) == 0:
        return_val = []
    else:
        return_val = numeral_decomp_builder(rem_int[0], int_sign, rem_int[1:], order + 1)
    if order == 0:
        return ['add', int_sign, int_in] + return_val
    add_term = ['add'] if len(rem_int) > 0 else []
    coeff_term = ['mul', int_sign, int_in] if (int(int_in) != 1 or int_sign != 'INT+') else []
    if order == 1:
        return add_term + coeff_term + ['INT+', '10'] + return_val
    else:
        return add_term + coeff_term + ['pow', 'INT+', '10'] + write_int(order, numeral_decomp=True) + return_val


def write_infix(token, args):
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
    elif token.startswith('INT'):
        return f'{token[-1]}{args[0]}'
    else:
        raise NameError(f"Unknown token in prefix expression: {token}, with arguments {args}")


def parse_int(lst, base=10):
    """
    Parse a list that starts with an integer.
    Return the integer value, and the position it ends in the list.
    """
    val = 0
    if not (base >= 2 and lst[0] in ['INT+', 'INT-'] or base <= -2 and lst[0] == 'INT'):
        raise TypeError(f"Invalid integer in prefix expression")
    i = 0
    for x in lst[1:]:
        if not (x.isdigit() or x[0] == '-' and x[1:].isdigit()):
            break
        val = val * base + int(x)
        i += 1
    if base > 0 and lst[0] == 'INT-':
        val = -val
    return val, i + 1
