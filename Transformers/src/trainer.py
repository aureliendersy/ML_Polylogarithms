# Adapted from https://github.com/facebookresearch/SymbolicMathematics under CC BY-NC 4.0

"""
Contains the training loop and various routines for efficient data loading
"""


import os
import io
import re
import sys
import shutil
from collections import OrderedDict
from math import floor
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from sympy import mathematica_code
import pathos.multiprocessing as mp
from src.envs.wolfram_utils import *

from .optim import get_optimizer
from .utils import to_cuda, TimeoutError

if torch.cuda.is_available():
    import apex


logger = getLogger()


class Trainer(object):

    EQUATIONS = {}

    def __init__(self, modules, env, params):
        """
        Initialize trainer.
        """
        # modules / params
        self.modules = modules
        self.params = params
        self.env = env

        # epoch / iteration size
        self.epoch_size = params.epoch_size
        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        # data iterators
        self.iterators = {}

        # set parameters
        self.set_parameters()

        # float16 / distributed (no AMP)
        assert params.amp >= 1 or not params.fp16
        assert params.amp >= 0 or params.accumulate_gradients == 1
        if params.multi_gpu and params.amp == -1:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            for k in self.modules.keys():
                self.modules[k] = nn.parallel.DistributedDataParallel(self.modules[k], device_ids=[params.local_rank], output_device=params.local_rank, broadcast_buffers=True)

        # set optimizers
        self.set_optimizers()

        # float16 / distributed (AMP)
        if params.amp >= 0:
            self.init_amp()
            if params.multi_gpu:
                logger.info("Using apex.parallel.DistributedDataParallel ...")
                for k in self.modules.keys():
                    self.modules[k] = apex.parallel.DistributedDataParallel(self.modules[k], delay_allreduce=True)

        # stopping criterion used for early stopping
        if params.stopping_criterion != '':
            split = params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == '_':
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(',') if m != '']
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.stats = OrderedDict(
            [('processed_e', 0)] +
            [('processed_w', 0)] +
            sum([
                [(x, []), (f'{x}-AVG-STOP-PROBS', [])]
                for x in env.TRAINING_TASKS
            ], [])
        )
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

        # file handler to export data
        if params.export_data:
            assert params.reload_data == ''
            params.export_path_prefix = os.path.join(params.dump_path, 'data.prefix')
            params.export_path_infix = os.path.join(params.dump_path, 'data.infix')

            # We export the data in MMA format to compute the appropriate symbol
            if params.export_mma:
                self.export_mma = True
                params.export_path_mma = os.path.join(params.dump_path, 'data.mma')
                self.file_handler_mma = io.open(params.export_path_mma, mode='a', encoding='utf-8')
                logger.info(f"Data will be stored in Mathematica in: {params.export_path_mma} ...")
            else:
                self.export_mma = False

            self.file_handler_prefix = io.open(params.export_path_prefix, mode='a', encoding='utf-8')
            self.file_handler_infix = io.open(params.export_path_infix, mode='a', encoding='utf-8')
            logger.info(f"Data will be stored in prefix in: {params.export_path_prefix} ...")
            logger.info(f"Data will be stored in infix in: {params.export_path_infix} ...")

        # reload exported data
        if params.reload_data != '':
            assert params.export_data is False
            s = [x.split(',') for x in params.reload_data.split(';') if len(x) > 0]
            assert len(s) >= 1 and all(len(x) == 4 for x in s) and len(s) == len(set([x[0] for x in s]))
            self.data_path = {task: (train_path, valid_path, test_path) for task, train_path, valid_path, test_path in s}
            assert all(all(os.path.isfile(path) for path in paths) for paths in self.data_path.values())
            for task in self.env.TRAINING_TASKS:
                assert (task in self.data_path) == (task in params.tasks)
        else:
            self.data_path = None

        # create data loaders
        if not params.eval_only:
            if params.env_base_seed < 0:
                params.env_base_seed = np.random.randint(1_000_000_000)
            self.dataloader = {
                task: iter(self.env.create_train_iterator(task, params, self.data_path))
                for task in params.tasks
            }

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend([(k, p) for k, p in v.named_parameters() if p.requires_grad])
        self.parameters['model'] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizers(self):
        """
        Set optimizers.
        """
        params = self.params
        self.optimizers = {}
        self.optimizers['model'] = get_optimizer(self.parameters['model'], params.optimizer)
        logger.info("Optimizers: %s" % ", ".join(self.optimizers.keys()))

    def init_amp(self):
        """
        Initialize AMP optimizer.
        """
        params = self.params
        assert params.amp == 0 and params.fp16 is False or params.amp in [1, 2, 3] and params.fp16 is True
        mod_names = sorted(self.modules.keys())
        opt_names = sorted(self.optimizers.keys())
        modules, optimizers = apex.amp.initialize(
            [self.modules[k] for k in mod_names],
            [self.optimizers[k] for k in opt_names],
            opt_level=('O%i' % params.amp)
        )
        self.modules = {k: module for k, module in zip(mod_names, modules)}
        self.optimizers = {k: optimizer for k, optimizer in zip(opt_names, optimizers)}

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            # exit()

        params = self.params

        # optimizers
        names = self.optimizers.keys()
        optimizers = [self.optimizers[k] for k in names]

        # regular optimization
        if params.amp == -1:
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                for name in names:
                    clip_grad_norm_(self.parameters[name], params.clip_grad_norm)
            for optimizer in optimizers:
                optimizer.step()

        # AMP optimization
        else:
            if self.n_iter % params.accumulate_gradients == 0:
                with apex.amp.scale_loss(loss, optimizers) as scaled_loss:
                    scaled_loss.backward()
                if params.clip_grad_norm > 0:
                    for name in names:
                        clip_grad_norm_(apex.amp.master_params(self.optimizers[name]), params.clip_grad_norm)
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                with apex.amp.scale_loss(loss, optimizers, delay_unscale=True) as scaled_loss:
                    scaled_loss.backward()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % 20 != 0:
            return

        s_iter = "%7i - " % self.n_total_iter
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k.upper().replace('_', '-'), np.mean(v))
            for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rates
        s_lr = ""
        for k, v in self.optimizers.items():
            s_lr = s_lr + (" - %s LR: " % k) + " / ".join("{:.4e}".format(group['lr']) for group in v.param_groups)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} equations/s - {:8.2f} words/s - ".format(
            self.stats['processed_e'] * 1.0 / diff,
            self.stats['processed_w'] * 1.0 / diff
        )
        self.stats['processed_e'] = 0
        self.stats['processed_w'] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def save_checkpoint(self, name, include_optimizers=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics,
            'best_stopping_criterion': self.best_stopping_criterion,
            'params': {k: v for k, v in self.params.__dict__.items()},
        }

        for k, v in self.modules.items():
            logger.warning(f"Saving {k} parameters ...")
            data[k] = v.state_dict()

        if include_optimizers:
            for name in self.optimizers.keys():
                logger.warning(f"Saving {name} optimizer ...")
                data[f'{name}_optimizer'] = self.optimizers[name].state_dict()

        torch.save(data, path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == '':
                return
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location='cpu')

        # reload model parameters
        for k, v in self.modules.items():
            v.load_state_dict(data[k])

        # reload optimizers
        for name in self.optimizers.keys():
            # AMP checkpoint reloading is buggy, we cannot reload optimizers
            # instead, we only reload current iterations / learning rates
            if self.params.amp == -1:
                logger.warning(f"Reloading checkpoint optimizer {name} ...")
                self.optimizers[name].load_state_dict(data[f'{name}_optimizer'])
            else:
                logger.warning(f"Not reloading checkpoint optimizer {name}.")
                for group_id, param_group in enumerate(self.optimizers[name].param_groups):
                    if 'num_updates' not in param_group:
                        logger.warning(f"No 'num_updates' for optimizer {name}.")
                        continue
                    logger.warning(f"Reloading 'num_updates' and 'lr' for optimizer {name}.")
                    param_group['num_updates'] = data[f'{name}_optimizer']['param_groups'][group_id]['num_updates']
                    param_group['lr'] = self.optimizers[name].get_lr_for_step(param_group['num_updates'])

        # reload main metrics
        self.epoch = data['epoch'] + 1
        self.n_total_iter = data['n_total_iter']
        self.best_metrics = data['best_metrics']
        self.best_stopping_criterion = data['best_stopping_criterion']
        logger.warning(f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ...")

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if self.params.save_periodic > 0 and self.epoch % self.params.save_periodic == 0:
            self.save_checkpoint('periodic-%i' % self.epoch)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning("Metric \"%s\" not found in scores!" % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_checkpoint('best-%s' % metric)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None and (self.params.is_master or not self.stopping_criterion[0].endswith('_mt_bleu')):
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
            else:
                logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info("Stopping criterion has been below its best value for more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                if self.params.multi_gpu and 'SLURM_JOB_ID' in os.environ:
                    os.system('scancel ' + os.environ['SLURM_JOB_ID'])
                exit()
        self.save_checkpoint('checkpoint')
        self.epoch += 1

    def get_batch(self, task):
        """
        Return a training batch for a specific task.
        """
        try:
            batch = next(self.dataloader[task])
        except Exception as e:
            logger.error("An unknown exception of type {0} occurred in line {1} when fetching batch. "
                         "Arguments:{2!r}. Restarting ...".format(type(e).__name__, sys.exc_info()[-1].tb_lineno, e.args))
            if self.params.is_slurm_job:
                if int(os.environ['SLURM_PROCID']) == 0:
                    logger.warning("Requeuing job " + os.environ['SLURM_JOB_ID'])
                    os.system('scontrol requeue ' + os.environ['SLURM_JOB_ID'])
                else:
                    logger.warning("Not the master process, no need to requeue.")
            raise

        return batch

    def export_data(self, task):
        """
        Export data to the disk.
        """
        env = self.env
        (x1, len1), (x2, len2), _ = self.get_batch(task)
        missed_eq = 0
        for i in range(len(len1)):

            # prefix
            prefix1 = [env.id2word[wid] for wid in x1[1:len1[i] - 1, i].tolist()]
            prefix2 = [env.id2word[wid] for wid in x2[1:len2[i] - 1, i].tolist()]
            # infix
            infix1 = env.prefix_to_infix(prefix1)[1:-1]
            infix2 = env.prefix_to_infix(prefix2)[1:-1]

            try:
                # When we generate data and we require the symbol it is useful to give it in MMA parsed format
                if task == 'symbol_int':
                    mma_str = mathematica_code(env.infix_to_sympy(infix2, check_stack=False))

                    if self.export_mma:

                        # If we want to compute the symbol in Python directly (not recommended)
                        if self.params.compute_symbol_on_site:
                            str_symbol = compute_symbol_wolfram(env.session, mma_str, symbol_factor=False)
                            sympy_expr_symbol = parse_expr(str_symbol)
                            prefix1 = env.sympy_to_prefix(sympy_expr_symbol)
                            infix1 = env.prefix_to_infix(prefix1)[1:-1]

                        self.file_handler_mma.write(f'{mma_str}\n')
                        self.file_handler_mma.flush()

                # save
                prefix1_str = ' '.join(prefix1)
                prefix2_str = ' '.join(prefix2)
                infix1_str = re.sub(r'(?<![a-zA-Z0-9])\(([a-zA-Z0-9]+)\)(?<![a-zA-Z0-9])', r'\1', infix1)
                infix2_str = re.sub(r'(?<![a-zA-Z0-9])\(([a-zA-Z0-9]+)\)(?<![a-zA-Z0-9])', r'\1', infix2)

                self.file_handler_prefix.write(f'{prefix1_str}\t{prefix2_str}\n')
                self.file_handler_infix.write(f'{infix1_str}\t{infix2_str}\n')
                self.file_handler_prefix.flush()
                self.file_handler_infix.flush()
                self.EQUATIONS[(prefix1_str, prefix2_str)] = self.EQUATIONS.get((prefix1_str, prefix2_str), 0) + 1

            except TimeoutError:
                logger.info("Unable to compute symbol -> {}".format(mma_str))
                missed_eq += 1
                continue

        # number of processed sequences / words
        self.n_equations += self.params.batch_size
        self.n_equations -= missed_eq
        self.stats['processed_e'] += len1.size(0)
        self.stats['processed_w'] += (len1 + len2 - 2).sum().item()

    def import_symbol_output(self, filepath, mma_rewrite=True, multiprocess_read=False, num_workers=1):
        """Read the result of the symbol computation done directly in Mathematica
        Add it to the relevant files. Supports multiprocessing"""

        # Start by defining the relevant file paths
        file_symbol = open(filepath, "r")
        symbol_lines = file_symbol.readlines()

        if mma_rewrite:
            file_symbol_mma = open(filepath.replace('.txt', '_mma.txt'), "r")
            symbol_lines_mma = file_symbol_mma.readlines()
        else:
            symbol_lines_mma = None

        new_symbol_path = filepath.replace('.txt', '_new.txt')
        new_symbol_path_mma = filepath.replace('.txt', '_mma_new.txt')

        prefix_file = open(self.params.export_path_prefix, "r")
        lines_prefix = prefix_file.readlines()

        infix_file = open(self.params.export_path_infix, "r")
        lines_infix = infix_file.readlines()

        assert (len(lines_prefix) == len(lines_infix))
        logger.info("Loading {} symbols to be inserted back".format(str(len(symbol_lines))))

        # Use multiprocessing to read the files
        if multiprocess_read:

            new_prefix_path_hid = self.params.export_path_prefix.replace('.prefix', '_new.prefix')
            new_infix_path_hid = self.params.export_path_infix.replace('.infix', '_new.infix')
            new_symbol_path_hid = new_symbol_path.replace('.txt', '_hid.txt')

            if mma_rewrite:
                new_symbol_path_mma_hid = new_symbol_path_mma.replace('.txt', '_hid.txt')
            else:
                new_symbol_path_mma_hid = None

            outpaths = [new_prefix_path_hid, new_infix_path_hid, new_symbol_path_hid, new_symbol_path_mma_hid]

            total_batch_size_lim = floor(len(symbol_lines) / 100000)

            batched_symbols = split_list(symbol_lines, max(num_workers * total_batch_size_lim, num_workers))
            if symbol_lines_mma is None:
                batched_symbols_mma = [None for _ in range(len(batched_symbols))]
            else:
                batched_symbols_mma = split_list(symbol_lines_mma, max(num_workers * total_batch_size_lim, num_workers))
            batched_prefix = split_list(lines_prefix, max(num_workers * total_batch_size_lim, num_workers))
            batched_infix = split_list(lines_infix, max(num_workers * total_batch_size_lim, num_workers))
            num_batches = max(num_workers * total_batch_size_lim, num_workers)

            args = ((self.params, batched_symbols[i], batched_symbols_mma[i], batched_prefix[i], batched_infix[i], mma_rewrite, i*100/num_batches)
                    for i in range(max(num_workers * total_batch_size_lim, num_workers)))

            pool = mp.ProcessingPool(nodes=num_workers)
            ft = lambda x: add_symbol_info_temp_file(*x)
            outpaths_batched = pool.map(ft, args)

            # To remove the last line, OS agnostic
            remove_chars = len(os.linesep)

            for i, path_file in enumerate(outpaths):
                if path_file is not None:
                    with open(path_file, 'wb') as wfd:
                        for f in outpaths_batched:
                            with open(f[i], 'rb') as fd:
                                shutil.copyfileobj(fd, wfd)
                                wfd.write(b"\n")
                            wfd.truncate(wfd.tell() - remove_chars)
                            remove(f[i])
        else:

            new_prefix_path_hid, new_infix_path_hid, new_symbol_path_hid, new_symbol_path_mma_hid\
                = add_symbol_info_temp_file(self.params, symbol_lines, symbol_lines_mma, lines_prefix, lines_infix, mma_rewrite, 100)

        file_symbol.close()

        # Copy the file permissions from the old file to the new file
        copymode(self.params.export_path_prefix, new_prefix_path_hid)
        copymode(self.params.export_path_infix, new_infix_path_hid)
        copymode(filepath, new_symbol_path_hid)
        if mma_rewrite:
            copymode(filepath.replace('.txt', '_mma.txt'), new_symbol_path_mma_hid)

        # Remove original file (prefix and infix)
        remove(self.params.export_path_prefix)
        remove(self.params.export_path_infix)

        # Move new file
        move(new_prefix_path_hid, self.params.export_path_prefix)
        move(new_infix_path_hid, self.params.export_path_infix)
        move(new_symbol_path_hid, new_symbol_path)
        if mma_rewrite:
            move(new_symbol_path_mma_hid, new_symbol_path_mma)

    def extend_symbol_output(self, filepath, repet_number, info_path=None):
        """If we scrambled the symbol multiple times (corresponding to the same function)
        we can add it back and match it to the relevant function"""
        prefix_filepath = self.params.export_path_prefix
        infix_filepath = self.params.export_path_infix
        add_extended_symbol_info(self.env, filepath, prefix_filepath, infix_filepath, repet_number, repet_info=info_path)

    def enc_dec_step(self, task):
        """
        Encoding / decoding step.
        """
        params = self.params
        encoder, decoder = self.modules['encoder'], self.modules['decoder']
        encoder.train()
        decoder.train()

        # batch
        (x1, len1), (x2, len2), _ = self.get_batch(task)

        # target words to predict
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
        y = x2[1:].masked_select(pred_mask[:-1])
        assert len(y) == (len2 - 1).sum().item()

        # cuda
        x1, len1, x2, len2, y = to_cuda(x1, len1, x2, len2, y)

        # forward / loss
        encoded = encoder('fwd', x=x1, lengths=len1, causal=False)
        decoded = decoder('fwd', x=x2, lengths=len2, causal=True, src_enc=encoded.transpose(0, 1), src_len=len1)
        _, loss = decoder('predict', tensor=decoded, pred_mask=pred_mask, y=y, get_scores=False)
        self.stats[task].append(loss.item())

        # optimize
        self.optimize(loss)

        # number of processed sequences / words
        self.n_equations += params.batch_size
        self.stats['processed_e'] += len1.size(0)
        self.stats['processed_w'] += (len1 + len2 - 2).sum().item()
