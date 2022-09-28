import numpy as np
import json

import src
from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import AttrDict, initialize_exp
from src.envs import build_env
from src.model import build_modules, check_model_params
from src.envs.wolfram_utils import *

from src.trainer import Trainer
from src.evaluator import Evaluator


np.seterr(all='raise')


def main(params):

    init_distributed_mode(params)
    logger = initialize_exp(params)
    init_signal_handler()

    src.utils.CUDA = not params.cpu

    env = build_env(params)

    # Open a wolfram session if necessary
    if params.compute_symbol_on_site or (params.symbol_check and params.eval_only):
        env.session = start_wolfram_session(kernel_path=env.kernel_path, lib_path=env.lib_path)

    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    evaluator = Evaluator(trainer)

    # evaluation
    if params.eval_only:
        scores = evaluator.run_all_evals()
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))

        # If we need to close the Wolfram session
        if params.compute_symbol_on_site or (params.symbol_check and params.eval_only):
            end_wolfram_session(env.session)

        exit()


if __name__ == '__main__':

    parameters = AttrDict({

        # Name
        'exp_name': 'eval_symbol',
        'dump_path': 'dump_path',
        'exp_id': 'weight2',
        'save_periodic': 0,
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
        'max_ops': 8,
        'max_terms_transc': 3,
        'max_ops_G': 15,
        'clean_prefix_expr': True,
        'rewrite_functions': '',
        'operators': 'add:20,sub:20,mul:10,div:15,pow2:4,pow3:1,pow4:0.0',

        # model parameters
        'emb_dim': 512,
        'n_enc_layers': 3,
        'n_dec_layers': 3,
        'n_heads': 4,
        'dropout': 0,
        'attention_dropout': 0,
        'sinusoidal_embeddings': False,
        'share_inout_emb': True,
        'reload_model': 'model_path',

        # Trainer param
        'export_data': False,
        'export_mma': False,
        'compute_symbol_on_site': False, # Best to do it with an external Mathematica script
        'reload_data': 'symbol_int,path_train,path_valid,path_test',
        'reload_size': '',
        'epoch_size': 10000,
        'max_epoch': 400,
        'amp': 2,  # Set to 2 if running on cluster and training
        'fp16': True, # Set to True if running on cluster and training
        'accumulate_gradients': 1,
        'optimizer': "adam,lr=0.0001",
        'clip_grad_norm': 5,
        'stopping_criterion': '',
        'validation_metrics': 'valid_func_simple_acc',
        'reload_checkpoint': '',
        'env_base_seed': -1,
        'batch_size': 1,
        'same_nb_ops_per_batch': False,

        # Evaluation
        'eval_only': True,
        'numerical_check': False,
        'symbol_check': False,
        'eval_verbose': 2,
        'eval_verbose_print': True,
        'beam_eval': True,
        'beam_size': 10,
        'beam_length_penalty': 1,
        'beam_early_stopping': True,

        # SLURM/GPU param
        'cpu': False,  # Set to False if using a GPU
        'local_rank': -1,
        'master_port': -1,
        'num_workers': 1,
        'debug_slurm': False,

        # Wolfram Client parameters (path to the wolfram kernel and the PolyLogTools library)
        'kernel_path': None,
        'lib_path': None,
    })

    check_model_params(parameters)
    main(parameters)
