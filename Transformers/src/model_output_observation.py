"""

Study the output of the model with some handpicked symbol examples
Used for a qualitative check of the output

"""

from statistics import mean
import torch
import src
from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import AttrDict, to_cuda, initialize_exp
from src.envs import build_env
from src.model import build_modules, check_model_params
from src.envs.wolfram_utils import *
import sympy as sp


def test_model_on_set(eqs_set, environment, module_transfo, verbose=True, file_type='RLdata', num_eqs_read=100):
    """
    Test a given transformer model on a dataset
    Dataset can be generated either with the RL data generation procedure or with the the prefix one
    :param eqs_set:
    :param environment:
    :param module_transfo:
    :param verbose:
    :param file_type:
    :param num_eqs_read:
    :return:
    """
    print('Test transformer on model data :')
    print('Data from : {}'.format(eqs_set))

    # Read the data from the RL source (in str(sympy) form)
    if file_type == 'RLdata':
        initial_set_file = open(eqs_set, "r")
        initial_eqs = initial_set_file.readlines()
        simple_expr = [eqs[:-1].split(': ')[-1] for eqs in initial_eqs[1::4]]
        scr_expr = [eqs[:-1].split(': ')[-1] for eqs in initial_eqs[2::4]]

    # Read the data in prefix form
    elif file_type == 'prefix_file':
        with open(eqs_set) as file_read:
            first_prefix = [next(file_read) for _ in range(num_eqs_read)]
            simple_expr = [eq.replace('|', '\t').split('\t')[2][:-1].split(' ') for eq in first_prefix]
            scr_expr = [eq.replace('|', '\t').split('\t')[1].split(' ') for eq in first_prefix]

    else:
        with open(eqs_set) as file_read:
            first_eqs = [next(file_read) for _ in range(num_eqs_read)]
            simple_expr = [eq.split('\t')[1][:-1] for eq in first_eqs]
            scr_expr = [eq.split('\t')[0] for eq in first_eqs]

    print('Loaded {} equations from the disk'.format(len(scr_expr)))

    sucesses = []

    # Go through the set and look how many times we match the desired target
    for j, simple_eq in enumerate(simple_expr):
        if j % 10 == 0:
            print('Did {} equations'.format(j))
        print('Eq {}'.format(str(j)))
        sucesses.append(test_model(environment, module_transfo, scr_expr[j], source_set=file_type,
                                   target_equation=simple_eq, verbose=verbose))

    print('In our beam search we solved {} equations '.format(len([num for num in sucesses if num is not None])))
    print('In our beam search we solved {} equations first try '.format(len([num for num in sucesses if num == 1])))
    print('Needed on average {} tries for solving '.format(mean([num for num in sucesses if num is not None])))


def test_model(environment, module_transfo, input_equation, source_set='Unspecified', target_equation=None,
               verbose=True):
    """
    Test the capacity of the transformer model to resolve a given input
    :param environment:
    :param module_transfo:
    :param input_equation:
    :param source_set:
    :param target_equation:
    :param verbose:
    :return:
    """

    # load the transformer
    encoder = module_transfo['encoder']
    decoder = module_transfo['decoder']
    encoder.eval()
    decoder.eval()

    # Convert our input
    if source_set == 'prefix_file':
        x1_prefix = input_equation
        f = str(environment.infix_to_sympy(environment.prefix_to_infix(x1_prefix)))
    else:
        f = sp.parse_expr(input_equation)
        f_prefix = environment.sympy_to_prefix(f)
        x1_prefix = f_prefix
    x1 = torch.LongTensor([environment.eos_index] + [environment.word2id[w] for w in x1_prefix] + [environment.eos_index]).view(-1, 1)
    len1 = torch.LongTensor([len(x1)])
    x1, len1 = to_cuda(x1, len1)

    if source_set == 'prefix_file':
        x2_prefix = target_equation
        f2 = str(environment.infix_to_sympy(environment.prefix_to_infix(x2_prefix)))
        target_equation = f2

    # cuda
    x1, len1 = to_cuda(x1, len1)

    # forward
    encoded = encoder('fwd', x=x1, lengths=len1, causal=False)

    # Beam decoding
    beam_size = params.beam_size
    with torch.no_grad():
        _, _, beam = decoder.generate_beam(encoded.transpose(0, 1), len1, beam_size=beam_size, length_penalty=1,
                                           early_stopping=1, max_len=512)
        assert len(beam) == 1
    hypotheses = beam[0].hyp
    assert len(hypotheses) == beam_size

    if verbose:
        print(f"Input function f: {f}")
        print("")
        if target_equation is not None:
            print(f"Target function f: {target_equation}")
            print("")

    first_valid_num = None

    # Print out the scores and the hypotheses
    for num, (score, sent) in enumerate(sorted(hypotheses, key=lambda y: y[0], reverse=True)):

        # parse decoded hypothesis
        ids = sent[1:].tolist()  # decoded token IDs
        tok = [environment.id2word[wid] for wid in ids]  # convert to prefix

        try:
            hyp = environment.prefix_to_infix(tok)  # convert to infix
            hyp = environment.infix_to_sympy(hyp)  # convert to SymPy

            # When integrating the symbol
            if 'symbol_int' in params.tasks:
                matches, remaining_diff = compare_symbol_function(hyp, sp.parse_expr(input_equation.replace('citi', 'CiTi')),
                                                                  environment.session)

            # When simplifying a polylogarithmic expression in functional form
            else:
                remaining_diff = sp.simplify(
                    hyp - environment.infix_to_sympy(environment.prefix_to_infix(environment.sympy_to_prefix(sp.parse_expr(target_equation)))),
                    seconds=1)
                matches = remaining_diff == 0

            res = "OK" if matches else "NO"
            remain = "" if matches else " | {} remaining".format(remaining_diff)

            if matches and first_valid_num is None:
                first_valid_num = num + 1

        except:
            res = "INVALID PREFIX EXPRESSION"
            hyp = tok
            remain = ""

        if verbose:
            # print result
            print("%.5f  %s  %s %s" % (score, res, hyp, remain))

    if verbose:
        if first_valid_num is None:
            print('Could not solve')
        else:
            print('Solved in beam search')
        print("")
        print("")

    return first_valid_num


if __name__ == '__main__':

    # Example with integrating the symbol (don't need to precise the target, we can check it with MMA)
    input_eq = 'citi(x**2+x+1,x)'
    target_eq = None

    # Set the parameters (e.g model we want to look at)
    params = AttrDict({

        # Name
        'exp_name': 'Test_function',
        'dump_path': 'scratch_folder',
        'exp_id': 'testing',
        'tasks': 'symbol_int',
        'task_arg': 2,

        # environment parameters
        'env_name': 'char_sp',
        'int_base': 10,
        'balanced': False,
        'numeral_decomp': True,
        'positive': True,
        'precision': 10,
        'n_variables': 1,
        'n_coefficients': 0,
        'leaf_probs': '0.75,0,0.25,0',
        'frac_probs': '0.5, 0.25,0.25',
        'proba_arg_repeat': 0.5,
        'skew_basis_proba': 1,
        'max_len': 512,
        'max_int': 5,
        'max_int_tot': 10,
        'max_ops': 5,
        'max_terms_transc': 3,
        'max_ops_G': 15,
        'clean_prefix_expr': True,
        'rewrite_functions': '',
        'operators': 'add:20,sub:20,mul:10,div:15,pow2:4,pow3:1,pow4:0.0',
        'emb_dim': 512,
        'n_enc_layers': 3,
        'n_dec_layers': 3,
        'n_heads': 4,
        'dropout': 0,
        'attention_dropout': 0,
        'sinusoidal_embeddings': False,
        'share_inout_emb': True,
        'reload_model': '/Users/aurelien/Documents/Projects/Symbolic_ML/dumped/Test_train_symbol/weight2_no_scr/checkpoint.pth',
        'beam_size': 20,
        'cpu': True,
        'local_rank': -1,
        'master_port': -1,
        'kernel_path': None,
        'lib_path': None,
        'numerical_check': False,
        'symbol_check': False,
        })
    check_model_params(params)
    src.utils.CUDA = not params.cpu

    # Start the logger
    init_distributed_mode(params)
    logger = initialize_exp(params)
    init_signal_handler()

    # Load the model and environment
    env = build_env(params)

    modules = build_modules(env, params)
    x = env.local_dict['x']

    # start the wolfram session
    if 'symbol_int' in params.tasks:
        env.session = start_wolfram_session()

    first_num = test_model(env, modules, input_eq, verbose=True, target_equation=target_eq)
    print(first_num)

    if 'symbol_int' in params.tasks:
        env.session.stop()
