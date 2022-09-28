"""
Deterministic classical algorithms used to guide the search tree
"""

import time
import numpy as np
import sympy as sp
from copy import deepcopy
from sympy import poly, polylog

x = sp.Symbol('x')
DILOGS_IDS = {'dupl': {x**2: sp.Rational(1, 2), -x: -1},
              'refl': {1-x: -1},
              'inv': {1/x: -1}}


def canonical_kappa(*args):
    """
    Helper method for writting a canonical form
    :param args:
    :return:
    """
    kappa = 0
    i = 0
    for arg in args:
        kappa += 10**i * (abs(arg) + (abs(arg)-arg)/(max(4*abs(arg), 1)))
        i += 1
    return abs(kappa)


def inversion(polylog_term):
    order, arg = polylog_term.args
    return polylog(order, sp.cancel(1/arg))


def reflection(polylog_term):
    order, arg = polylog_term.args
    return polylog(order, sp.cancel(1-arg))


def generate_representatives(polylog_term):
    """
    Generate all dilog forms of the same complexity
    :param polylog_term:
    :return:
    """
    rep_list = [polylog_term]
    new_poly = polylog_term
    for _ in range(2):
        new_poly = inversion(new_poly)
        rep_list.append(new_poly)
        new_poly = reflection(new_poly)
        rep_list.append(new_poly)
    new_poly = inversion(new_poly)
    rep_list.append(new_poly)

    return rep_list


def extract_coeffs(polylog_term):
    """
    From the sympy expression extract the coefficients of the argument
    :param polylog_term:
    :return:
    """
    arg_poly = polylog_term.args[-1]
    num, denom = sp.fraction(sp.together(arg_poly))
    num_args = poly(num, x).all_coeffs()
    num_args.reverse()
    num_args_e = num_args + [0]*(3 - len(num_args))
    denom_args = poly(denom, x).all_coeffs()
    denom_args.reverse()
    denom_args_e = denom_args + [0]*(3 - len(denom_args))

    return num_args_e, denom_args_e


def find_canonical(polylog_term):
    """
    Find the canonical representation corresponding to a polylog term
    This form is picked within the ones that are deemed of equal complexity
    :param polylog_term:
    :return:
    """
    list_rep = generate_representatives(polylog_term)
    list_weighted = []
    for i, rep in enumerate(list_rep):
        num_c, denom_c = extract_coeffs(rep)
        f_denom = [denom_c[0]]
        kappa_t = canonical_kappa(*(f_denom + num_c + denom_c[1:]))
        list_weighted.append([kappa_t, rep, i])

    list_weighted.sort(key=lambda xt: xt[0])

    return list_weighted[0]


def extract_polylogs(expression):
    """
    Take a given sympy expression and return the list of arguments
    that contain a polylog

    :param expression:
    :return:

    """

    list_args = sp.Add.make_args(expression)
    poly_args = [arg for arg in list_args if arg.has(sp.polylog)]

    return poly_args


def construct_polylog_dict(expression, weight=2):
    """Construct a dictionnary with the polylogs in a given expression
      along with their rational multiplicative factor"""

    dict_polylogs = {}

    for polylog_term in extract_polylogs(expression):

        # Get the weight and arg
        polylog_info = sp.Mul.make_args(polylog_term)

        # If no multiplicative factor, it is just 1
        if len(polylog_info) == 1:
            polylog_fact = 1
            polylog_expr = polylog_info[-1]
        else:
            polylog_fact = sp.Mul.make_args(polylog_term)[0]
            polylog_expr = sp.Mul.make_args(polylog_term)[-1]

        # Check that we extract only the polylogs of the desired weight
        if polylog_expr.args[0] == weight:
            simple_arg = sp.cancel(polylog_expr.args[1])

            # Simplify the arguments before adding them to the dictionnary
            # Check that we don't have duplicate arguments
            if simple_arg not in dict_polylogs:
                dict_polylogs[simple_arg] = polylog_fact
            else:
                dict_polylogs[simple_arg] += polylog_fact

    return dict_polylogs


def construct_polylog_expression(polylog_dict, weight=2):
    """Reconstruct the mathematical expression from the dictionnary"""

    math_expr = 0

    # Read from the dictionnary to reconstruct the dilog part
    for poly_argument, poly_factor in polylog_dict.items():
        math_expr += poly_factor * sp.polylog(weight, poly_argument)

    return math_expr


def apply_identity(polylog_dict, id_dict, polylog_arg):
    """Apply the required identity on the polylog term given with the dict structure"""

    new_polylog = deepcopy(polylog_dict)
    mult_fact = new_polylog[polylog_arg]

    for id_arg, id_fact in id_dict.items():
        new_arg = sp.cancel(id_arg.subs(x, polylog_arg))

        if new_arg not in new_polylog:
            new_polylog[new_arg] = mult_fact * id_fact
        else:
            new_polylog[new_arg] += mult_fact * id_fact

            # Get rid of the key argument if the factor in the expression is 0
            if new_polylog[new_arg] == 0:
                del new_polylog[new_arg]

    del new_polylog[polylog_arg]

    return new_polylog


def polylog_complexity(polylog_dict, complex_ops=False):
    """"Compute the associated complexity"""

    # Take 10 per dilog term
    complexity = len(polylog_dict.keys()) * 10

    # Add the complexity of the individual arguments afterwards
    if complex_ops:
        for polylog_arg in polylog_dict.keys():
            complexity += sp.count_ops(polylog_arg)

    return complexity


def reduction_search(polylog_dict, max_iter, ids_list, best_retained=1, complex_ops=False):
    """Similar to greedy search for simplifying expressions of dilogs but can be more extensive in the search space
    Essentially we can keep the best trajectories in memory for best_retained > 1"""

    # Keep track of the best expression (for each iteration and globally)
    best_expr = [(polylog_dict, [])]
    global_min = (polylog_complexity(polylog_dict), polylog_dict)

    # Also keep track of the identities used to reconstruct the log terms
    global_min_ids = []

    # Keep track of the number of visited nodes
    nodes_visited = 0

    # Loop on the iterations
    for iter in range(max_iter):
        complexity_dict = {}

        # Loop on the number of expressions we retain on the head
        for expr in best_expr:
            polylog_expr = expr[0]

            # Loop on the arguments
            for polylog_arg in polylog_expr.keys():

                # Loop on the identities
                for id_name, identity in ids_list.items():

                    # Apply the identity and get the complexity
                    new_expr = apply_identity(polylog_expr, identity, polylog_arg)
                    complexity_comp = polylog_complexity(new_expr, complex_ops)
                    nodes_visited += 1

                    complexity_dict[(id_name, polylog_arg)] = (complexity_comp, new_expr, expr[-1])

                    # If we hit a new global minimum we save it
                    if complexity_comp < global_min[0]:
                        global_min = (complexity_comp, new_expr)
                        global_min_ids = deepcopy(expr[-1])
                        global_min_ids.append((id_name, polylog_arg))

                        if global_min == 0:
                            return global_min, global_min_ids, nodes_visited

                        # Retain the desired number of best expressions as the head for each iteration
        sorted_complexity_dict = sorted(complexity_dict.keys(), key=lambda xkey: complexity_dict[xkey][0])[:best_retained]
        best_expr = [(complexity_dict[identity_call][1], [*complexity_dict[identity_call][-1], identity_call])
                     for identity_call in sorted_complexity_dict]

    return global_min, global_min_ids, nodes_visited


def reduce_polylog_expression(expression, identity_list, tree_depth=None, beam_size=None, complex_ops=False):
    """
    Use a classical algorithm to reduce the polylog expression
    :param expression:
    :param identity_list:
    :param tree_depth:
    :param beam_size:
    :param complex_ops:
    :return:
    """
    # Convert the input to the dictionary format (keep arguments and factors)
    polylog_dict = construct_polylog_dict(expression)

    if tree_depth is None:
        iter_num = len(polylog_dict.keys()) * 4
    else:
        iter_num = tree_depth

    if beam_size is None:
        best_ret = len(polylog_dict.keys()) * 3
    else:
        best_ret = beam_size

    global_min, identity_used, nodes_visited = reduction_search(polylog_dict, iter_num, identity_list, best_ret,
                                                                complex_ops)

    return global_min[-1], identity_used, nodes_visited


def reconstruct_polylog_expression(polylog_dict, identity_list_full, identity_used, weight=2):
    """
    From the dict structure reconstruct the sympy expression
    :param polylog_dict:
    :param identity_list_full:
    :param identity_used:
    :param weight:
    :return:
    """
    full_expr = construct_polylog_expression(polylog_dict, weight)
    for identity in identity_used:
        full_expr += identity_list_full[identity[0]].subs(x, identity[1])

    return sp.simplify(full_expr)


def canonical_form_expression(expression):
    """
    Convert an expression into its canonical form
    :param expression:
    :return:
    """
    if expression == 0:
        return expression, 0

    expr_ret = 0
    ids_used = 0

    for polylog_term in extract_polylogs(expression):

        # Get the weight and arg
        polylog_info = sp.Mul.make_args(polylog_term)

        # If no multiplicative factor, it is just 1
        if len(polylog_info) == 1:
            polylog_fact = 1
            polylog_expr = polylog_info[-1]
        else:
            polylog_fact = sp.Mul.make_args(polylog_term)[0]
            polylog_expr = sp.Mul.make_args(polylog_term)[-1]
        kappa, expr, num_ids = find_canonical(polylog_expr)
        ids_used += 5
        expr_ret += polylog_fact * (-1)**num_ids * expr

    return expr_ret, ids_used


def apply_duplication(expression):
    """Duplication identity to be applied with the arguments in canonical form"""
    ids_used = 0
    expression_temp = 0
    in_num_poly = expression.count(polylog)

    for polylog_term in extract_polylogs(expression):

        # Get the weight and arg
        polylog_info = sp.Mul.make_args(polylog_term)

        # If no multiplicative factor, it is just 1
        if len(polylog_info) == 1:
            polylog_fact = 1
            polylog_expr = polylog_info[-1]
        else:
            polylog_fact = sp.Mul.make_args(polylog_term)[0]
            polylog_expr = sp.Mul.make_args(polylog_term)[-1]

        new_expr = 0

        new_term1 = polylog(polylog_expr.args[0], sp.cancel(-polylog_expr.args[-1]))
        kappa, expr, num_ids = find_canonical(new_term1)
        new_expr += polylog_fact * (-1)**(num_ids+1) * expr
        ids_used += num_ids

        new_term2 = polylog(polylog_expr.args[0], sp.cancel(polylog_expr.args[-1]**2))
        kappa, expr, num_ids = find_canonical(new_term2)
        new_expr += polylog_fact * sp.Rational(1, 2) * (-1)**num_ids * expr
        ids_used += num_ids

        expression_temp = expression.subs(polylog_term, new_expr)
        num_new_poly = expression_temp.count(polylog)

        if num_new_poly < in_num_poly:
            return expression_temp, ids_used

    return expression_temp, ids_used


def canonical_reduction(expression, tree_depth):
    """Use the duplication identity only to do a reduction"""
    canonical_form, ids_used = canonical_form_expression(expression)
    found_solution = canonical_form == 0
    depth = 0

    while depth < tree_depth and not found_solution:
        canonical_form, ids_use = apply_duplication(canonical_form)
        ids_used += ids_use

        if canonical_form == 0:
            return canonical_form, ids_used

        depth += 1

    return canonical_form, ids_used


def bfs(initial_eq, target_eq, max_levels=10, ids_list=DILOGS_IDS, max_time=60):
    """Do a naive Breadth First Search to solve the simplification tree
    Will quickly become slow as the number of level traversed increases"""
    timeout = time.time() + max_time
    polylog_expr = construct_polylog_dict(initial_eq)
    target_expr = construct_polylog_dict(target_eq)
    found_solution = False
    num_ids_used = 0
    levels_visited = 1
    last_level_expr = [polylog_expr]

    while not found_solution and levels_visited < max_levels and time.time() < timeout:
        new_level = []
        for expr_level in last_level_expr:
            # Loop on the arguments
            for polylog_arg in expr_level.keys():

                # Loop on the identities
                for id_name, identity in ids_list.items():
                    new_expr = apply_identity(expr_level, identity, polylog_arg)
                    new_level.append(new_expr)

                    num_ids_used += 1
                    found_solution = new_expr == target_expr
                    if found_solution:
                        return found_solution, num_ids_used, levels_visited

        last_level_expr = new_level
        levels_visited += 1
    return found_solution, num_ids_used, levels_visited


def smarter_bfs(initial_eq, target_eq, max_levels=10, ids_list=DILOGS_IDS, max_time=60):
    """Do a Breadth First Search but discard any nodes that cannot possibly lead to the
    solution in the number of remaining allocated moves"""
    timeout = time.time() + max_time
    polylog_expr = construct_polylog_dict(initial_eq)
    target_expr = construct_polylog_dict(target_eq)
    num_poly_target = len(target_expr.keys())
    found_solution = False
    num_ids_used = 0
    levels_visited = 1
    last_level_expr = [polylog_expr]

    while not found_solution and levels_visited <= max_levels and time.time() < timeout:
        new_level = []
        for expr_level in last_level_expr:
            # Loop on the arguments
            for polylog_arg in expr_level.keys():

                # Loop on the identities
                for id_name, identity in ids_list.items():
                    new_expr = apply_identity(expr_level, identity, polylog_arg)
                    # Check that the expression is not that far away that we cannot hope to solve it
                    # A duplication identity can hope to simplify at most 3 terms at once
                    if len(new_expr.keys()) < 3*(max_levels - levels_visited + 1) + num_poly_target:
                        # Also check if we have not yet encountered this expression
                        if new_expr not in new_level:
                            new_level.append(new_expr)
                    num_ids_used += 1
                    found_solution = new_expr == target_expr
                    if found_solution:
                        return found_solution, num_ids_used, levels_visited

        last_level_expr = new_level
        levels_visited += 1
    return found_solution, num_ids_used, levels_visited


def import_eqs_set(start_set_path, add_info=False):
    """
    Given a file path import the relevant equations and separate them into the scrambled equations and the goal
    :param start_set_path:
    :param add_info:
    :return:
    """
    print('On set :')
    initial_set_file = open(start_set_path, "r")
    initial_eqs = initial_set_file.readlines()

    simple_expr = [sp.sympify(eqs[:-1].split(': ')[-1]) for eqs in initial_eqs[1::4]]
    scr_expr = [sp.sympify(eqs[:-1].split(': ')[-1]) for eqs in initial_eqs[2::4]]

    if add_info:
        eq_info = [eqs for eqs in initial_eqs[0::4]]
        eq_info_complex = [(int(eq[eq.find(" scrambles") - 1]), int(eq[eq.find(" different terms") - 1])) for eq in
                           eq_info]
    else:
        eq_info_complex = None

    return scr_expr, simple_expr, eq_info_complex


def beam_search_network(scr_expr, model=None, env=None, beam_size=1, breadth_size=2, gamma=0.9):
    """
    Take a trained network as input and apply it to reduce a scrambled expression using the beam search approach
    Keep beam_size equations in memory and try out the best breadth_size actions
    :param scr_expr:
    :param model:
    :param env:
    :param beam_size:
    :param breadth_size:
    :param gamma:
    :return:
    """
    if model is None or env is None:
        raise ValueError('Need both a model and an initial environment for the beam search')

    if len(env.obs_rep.sympy_to_prefix(scr_expr)) > env.obs_rep.dim:
        return None, None, None

    # Initialize the environment using each new equation
    env.init_eq = scr_expr

    reward_tot, nodes_visited, nodes_visited_non_cyclic = 0, 0, 0
    steps_done_distrib = {'0': 0, "1": 0, '2': 0, '3': 0}
    obs = env.reset()

    # Initialize the beam head (observation, environment, number of actions taken, total reward)
    beam_head = [(obs, env, 0, 0)]

    for i in range(env.max_steps):
        beam_rwd = []
        beam_values = []
        beam_envs = []
        beam_obs = []
        beam_steps = []
        for environment in beam_head:

            action_distr = model.policy.get_distribution(
                model.policy.obs_to_tensor(environment[0])[0]).distribution.probs.detach().numpy()[0]

            retained_actions = (-action_distr).argsort()[0:breadth_size]

            for action in retained_actions:

                # Get a copy of the environment on which we will act with the policy
                new_env = deepcopy(environment[1])
                new_env.obs_rep.prefix_state = deepcopy(environment[1].obs_rep.prefix_state)

                # Need to pay special care to the tuple making up the arguments of the sympy expression
                env_args = environment[1].obs_rep.sp_state.args[:]
                new_env.obs_rep.sp_state = new_env.obs_rep.sp_state._new_rawargs(*env_args)

                # Act on the copied environment with the retained action
                obstemp, reward, donestemp, _ = new_env.step(int(action))
                nodes_visited += 1
                beam_rwd.append(environment[3] + reward * gamma ** i)
                steps_done_distrib[str(action)] += 1

                # If the action was non cyclic we count it as visiting a node
                if action != 2:
                    nodes_visited_non_cyclic += 1
                    beam_steps.append(environment[2]+1)
                else:
                    beam_steps.append(environment[2])

                # If we found the solution we can stop there
                if new_env.obs_rep.sp_state == 0:
                    return 0, environment[2]+1, nodes_visited_non_cyclic

                # Calculate the value of the new state with the value network
                valuetemp = model.policy.predict_values(model.policy.obs_to_tensor(obstemp)[0]).detach().numpy()[0, 0]
                expect_tot_rwd = reward * gamma**i + valuetemp + environment[3]

                # Add to the beam head the values and environments
                beam_envs.append(new_env)
                beam_obs.append(obstemp)
                beam_values.append(expect_tot_rwd)

        beam_head_sorted = sorted(zip(beam_values, beam_obs, beam_envs, beam_steps, beam_rwd),
                                  reverse=True, key=lambda y: y[0])
        beam_head = [beam[1:] for beam in beam_head_sorted[0:beam_size]]

    return beam_head[0][1].obs_rep.sp_state, beam_head[0][2], nodes_visited_non_cyclic
